import json
import os
import random
import time
from copy import deepcopy
from itertools import chain

import hydra
import pandas as pd
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from r_baseline.ai_dataset import AiDataset
    from r_baseline.ai_loader import AiCollator, show_batch
    from r_baseline.ai_model import AiModel
    from r_baseline.ai_optimizer import get_optimizer
    from utils.metric_utils import compute_metrics
    from utils.train_utils import (AverageMeter, as_minutes, get_lr,
                                   init_wandb, print_gpu_utilization,
                                   print_line, save_checkpoint,
                                   seed_everything)

except Exception as e:
    print(e)
    raise ImportError

pd.options.display.max_colwidth = 1000

# -------- Evaluation -------------------------------------------------------------#


def run_evaluation(cfg, model, valid_dl):
    model.eval()

    all_ids = []
    all_predictions = []
    all_truths = []

    progress_bar = tqdm(range(len(valid_dl)))
    for batch in valid_dl:
        batch_ids = batch["id"]
        batch_truths = list(map(int, batch["labels"].cpu().numpy().tolist()))

        with torch.no_grad():
            logits, _, _ = model(**batch)
            logits = logits.reshape(-1)
            batch_preds = torch.sigmoid(logits)
        batch_preds = batch_preds.cpu().numpy().tolist()

        all_ids.extend(batch_ids)
        all_predictions.extend(batch_preds)
        all_truths.extend(batch_truths)

        progress_bar.update(1)
    progress_bar.close()

    # compute metric
    eval_dict = compute_metrics(all_predictions, all_truths)

    result_df = pd.DataFrame()
    result_df["id"] = all_ids
    result_df["predictions"] = all_predictions
    result_df["truths"] = all_truths

    oof_df = deepcopy(result_df)
    oof_df = oof_df.rename(columns={"predictions": "generated"})
    oof_df = oof_df[["id", "generated"]].copy()

    to_return = {
        "scores": eval_dict,
        "result_df": result_df,
        "oof_df": oof_df,
    }

    return to_return


# -------- Main Function ---------------------------------------------------------#


@hydra.main(version_base=None, config_path="../conf/r_baseline", config_name="conf_r_baseline")
def run_training(cfg):
    # ------- Runtime Configs -----------------------------------------------------#
    print_line()
    if cfg.use_random_seed:
        seed = random.randint(401, 999)
        cfg.seed = seed

    print(f"setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    # ------- folder management --------------------------------------------------#
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    # ------- load data ----------------------------------------------------------#
    print_line()
    data_dir = cfg.input_data_dir

    # load query dataframe
    essay_df = pd.read_csv(os.path.join(data_dir, "train_essays.csv"))
    essay_df['fold'] = essay_df['text'].apply(lambda x: 'train' if random.random() >= 0.01 else 'valid')
    # fold_df = pd.read_csv(os.path.join(data_dir, "folds.csv"))
    # essay_df = essay_df.merge(fold_df, on="id", how="left")

    # ------- Data Split ----------------------------------------------------------------#

    train_df = essay_df[essay_df["fold"] == 'train'].copy()
    valid_df = essay_df[essay_df["fold"] != 'train'].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of validation data: {valid_df.shape}")

    # ------- Datasets ------------------------------------------------------------------#
    # The datasets for ranking
    # -----------------------------------------------------------------------------------#

    dataset_creator = AiDataset(cfg)

    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)
    tokenizer = dataset_creator.tokenizer

    # ------- data loaders ----------------------------------------------------------------#
    data_collector = AiCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )

    train_ds.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )

    # sort valid dataset for faster evaluation
    valid_ds = valid_ds.sort("input_length")

    valid_ds.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.train_bs,
        shuffle=True,
        collate_fn=data_collector,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.valid_bs,
        shuffle=False,
        collate_fn=data_collector,
    )

    print("data preparation done...")
    print_line()

    # ------- Wandb --------------------------------------------------------------------#
    if cfg.use_wandb:
        print("initializing wandb run...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg)

    # --- show batch -------------------------------------------------------------------#
    print_line()

    for b in train_dl:
        break
    show_batch(b, tokenizer, task='training')

    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task='validation')

    print_line()

    # ------- Config -------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    # ------- Model --------------------------------------------------------------------#
    print_line()
    print("creating the Sci-LLM Ranker models...")
    model = AiModel(cfg)
    print_line()

    # ------- Optimizer ----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")
    optimizer = get_optimizer(model, cfg)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.grad_accumulation
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------- Accelerator --------------------------------------------------------------#
    print_line()
    print("accelerator setup...")
    accelerator = Accelerator(mixed_precision='bf16')  # cpu = True

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl,
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.  # track recall@1000
    save_trigger = cfg.train_params.save_trigger

    patience_tracker = 0
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()

        # Training ------
        model.train()
        for step, batch in enumerate(train_dl):
            logits, loss, loss_dict = model(**batch)
            accelerator.backward(loss)

            # take optimizer and scheduler steps
            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())

                progress_bar.set_description(
                    f"STEP: {step+1:5}/{num_update_steps_per_epoch:5}. "
                    f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (current_iteration-1) % cfg.train_params.eval_frequency == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(cfg, model, valid_dl)

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]

                lb = scores_dict["lb"]

                print_line()
                et = as_minutes(time.time()-start_time)
                print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                print(f">>> Current LB (AUC) = {round(lb, 4)}")

                print_line()

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                if is_best:
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_best.csv"), index=False)
                else:
                    print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                    print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_last.csv"), index=False)

                # saving -----
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': current_iteration,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lb': lb,
                }

                if best_lb > save_trigger:
                    save_checkpoint(cfg, model_state, is_best=is_best)

                # logging ----
                if cfg.use_wandb:
                    wandb.log({"lb": lb}, step=current_iteration)
                    wandb.log({"best_lb": best_lb}, step=current_iteration)

                    # -- log scores dict
                    for k, v in scores_dict.items():
                        wandb.log({k: round(v, 4)}, step=current_iteration)

                    # --- log best scores dict
                    for k, v in best_dict.items():
                        wandb.log({k: round(v, 4)}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg_dict['train_params']['patience']:
                    print("stopping early")
                    model.eval()
                    return


if __name__ == "__main__":
    run_training()
