import argparse
import math
import os
# utils ---------------------------------------------------------------------------------#
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset
from omegaconf import OmegaConf
from tokenizers import (Tokenizer, models, normalizers, pre_tokenizers,
                        processors, trainers)
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, DebertaV2Config,
                          DebertaV2ForMaskedLM, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast, default_data_collator,
                          get_cosine_schedule_with_warmup)
from transformers.models.deberta_v2.modeling_deberta_v2 import \
    DebertaV2OnlyMLMHead
from transformers.trainer_pt_utils import get_parameter_names


def count_digits(text):
    return len(re.findall(r'\d', text))


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm%ds' % (m, s)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# tokenizer class -----------------------------------------------------------------------#


def tokenizer_test(tokenizer):
    print("=="*40)
    print(f"tokenizer len: {len(tokenizer)}")
    test_string = "This is a test \n\n!!"
    print(f"tokenizer test: {tokenizer.tokenize(test_string)}")
    print("=="*40)

# collator class ------------------------------------------------------------------------#


@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()

        # geometric distribution for spans
        geo_p, lower, upper = 0.15, 1, 8
        len_distrib = [geo_p * (1-geo_p)**(i - lower) for i in range(lower, upper + 1)]
        len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
        lens = list(range(lower, upper + 1))

        masked_indices = []

        for ex_labels in labels:
            mask_num = math.ceil(len(ex_labels) * self.mlm_probability)
            ex_mask = set()
            while len(ex_mask) < mask_num:
                span_len = np.random.choice(lens, p=len_distrib)
                anchor = np.random.choice(len(ex_labels))
                if anchor in ex_mask:
                    continue
                else:
                    left1, right1 = anchor, min(anchor + span_len, len(ex_labels))
                    for i in range(left1, right1):
                        if len(ex_mask) >= mask_num:
                            break
                        ex_mask.add(i)
            ex_mask_bool = [i in ex_mask for i in range(len(ex_labels))]
            masked_indices.append(ex_mask_bool)
        masked_indices = torch.tensor(masked_indices).bool()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        masked_indices = torch.logical_and(masked_indices, ~special_tokens_mask)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # pdb.set_trace()

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def show_batch(batch, tokenizer, num_examples=8, print_fn=print):
    print_fn('=='*40)
    num_examples = min(num_examples, len(batch['input_ids']))

    for i in range(num_examples):
        input_ids = batch['input_ids'][i]
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print_fn(f"input text:\n {input_text}")
        print_fn('=='*40)

# dataset class -------------------------------------------------------------------------#


def get_mlm_dataset(cfg, notes_df, tokenizer):
    notes_df = notes_df[['text']].copy()
    notes_df = notes_df.reset_index(drop=True)

    task_dataset = Dataset.from_pandas(notes_df)

    def tokenize_function(examples):
        result = tokenizer(examples['text'])
        return result

    tokenized_datasets = task_dataset.map(
        tokenize_function, batched=True, remove_columns=task_dataset.column_names
    )

    chunk_size = cfg.max_length

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size

        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    test_pct = cfg.test_pct

    max_train_examples = cfg.max_train_examples
    max_test_examples = int(max_train_examples * test_pct)

    test_size = int(len(lm_datasets) * test_pct)
    train_size = len(lm_datasets) - test_size

    test_size = min(test_size, max_test_examples)
    train_size = min(train_size, max_train_examples)

    downsampled_dataset = lm_datasets.train_test_split(
        train_size=train_size, test_size=test_size, seed=cfg.seed
    )
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=cfg.mask_probability
    )

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    downsampled_dataset["test"] = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )

    try:
        downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
                "masked_token_type_ids": "token_type_ids",
            }
        )
    except Exception as e:
        downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )

    return downsampled_dataset

# main ----------------------------------------------------------------------------------#


def main(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision='fp16',
    )

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit*50 + suffix)

    # set seed ----
    print_line()
    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.model_dir, exist_ok=True)
    print_line()

    # data ------------------------------------------------------------------------------#
    notes_df = pd.read_csv(cfg.train_data_path).rename(columns={'full_text': 'text', 'essay_id_comp': 'id'})

    if 'mix' in cfg.train_data_path:
        notes_df = notes_df[notes_df['generated'] == 1].copy()  # filter LLM essays
        notes_df = notes_df.reset_index(drop=True)
        accelerator.print(f"shape of df after keeping only LLM essays: {notes_df.shape}")
        notes_df = notes_df[~notes_df['text'].apply(lambda x: "Ġ" in x)].copy()
        notes_df = notes_df[notes_df['text'].apply(lambda x: count_digits(x) < 64)].copy()
        accelerator.print(f"shape of df after removing bad examples: {notes_df.shape}")
        notes_df = notes_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        # notes_df = notes_df.reset_index(drop=True)
        accelerator.print(f"final shape of df: {notes_df.shape}")

    if cfg.debug:
        n_debug = min(1024, len(notes_df))
        notes_df = notes_df.sample(n_debug, random_state=cfg.seed).reset_index(drop=True)
    notes_df = notes_df[['id', 'text']].copy()

    accelerator.print(f"shape of input text data: {notes_df.shape}")
    print_line()

    # tokenizer -------------------------------------------------------------------------#
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_model_name)

    # dataset ---------------------------------------------------------------------------#
    with accelerator.main_process_first():
        mlm_dataset = get_mlm_dataset(cfg, notes_df, tokenizer)

    # model------------------------------------------------------------------------------#
    # base_config = DebertaV2Config(
    #     attention_probs_dropout_prob=0.1,
    #     hidden_act="gelu",
    #     hidden_dropout_prob=0.1,
    #     hidden_size=768,
    #     initializer_range=0.02,
    #     intermediate_size=3072,
    #     max_position_embeddings=cfg.max_length,
    #     relative_attention=True,
    #     position_buckets=256,
    #     norm_rel_ebd="layer_norm",
    #     share_att_key=True,
    #     pos_att_type="p2c|c2p",
    #     layer_norm_eps=1e-7,
    #     max_relative_positions=-1,
    #     position_biased_input=False,
    #     num_attention_heads=12,
    #     num_hidden_layers=6,
    #     type_vocab_size=0,
    #     vocab_size=tokenizer.vocab_size,
    # )

    # model = DebertaV2ForMaskedLM(base_config)
    base_config = AutoConfig.from_pretrained(cfg.backbone_model_name)
    # base_config.update(
    #     {
    #         "vocab_size": len(tokenizer),
    #         # "max_position_embeddings": 1024,

    #     }
    # )

    model = AutoModelForMaskedLM.from_pretrained(cfg.backbone_model_name, config=base_config)
    accelerator.print(f"tokenizer len: {len(tokenizer)}")
    accelerator.print(f"model vocab size: {model.config.vocab_size}")
    # model.deberta.resize_token_embeddings(len(tokenizer))
    model.cls = DebertaV2OnlyMLMHead(base_config)
    # model.gradient_checkpointing_enable()

    # optimizer -------------------------------------------------------------------------#
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    accelerator.print("using bnb optimizer....")

    optimizer = bnb.optim.Adam8bit(
        optimizer_grouped_parameters, lr=cfg.lr,
    )

    # collator --------------------------------------------------------------------------#

    eval_dataset = deepcopy(mlm_dataset['test'])

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=cfg.mask_probability
    )

    batch_size = cfg.per_device_batch_size

    train_dataloader = DataLoader(
        mlm_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        mlm_dataset["test"],
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    # show training batch ---
    for batch in train_dataloader:
        break
    show_batch(batch, tokenizer, num_examples=4, print_fn=accelerator.print)

    accelerator.print(f"Train dataset size: {len(mlm_dataset['train'])}")
    accelerator.print(f"Test dataset size: {len(mlm_dataset['test'])}")
    print_line()

    # prepare ---------------------------------------------------------------------------#
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    print_line()

    # scheduler -------------------------------------------------------------------------#
    print_line()
    num_epochs = cfg.num_train_epochs
    grad_accumulation_steps = cfg.gradient_accumulation_steps
    warmup_pct = cfg.warmup_pct

    num_update_steps_per_epoch = len(train_dataloader)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    accelerator.wait_for_everyone()

    # training --------------------------------------------------------------------------#
    start_time = time.time()
    current_iteration = 0

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()
        model.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)  # added gradient clip
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())
            # --
            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

            # Evaluation ----
            if (accelerator.sync_gradients) & (current_iteration % cfg.eval_frequency == 0):
                model.eval()
                losses = []

                n_correct = 0
                n_total = 0

                for _, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                        tok_preds = torch.max(outputs['logits'], dim=-1)[1]
                        curr = torch.masked_select(tok_preds == batch['labels'], batch['labels'] > -100).sum()
                        tot = torch.masked_select(tok_preds == batch['labels'], batch['labels'] > -100).size(0)
                        n_correct += curr
                        n_total += tot

                    loss = outputs.loss
                    losses.append(accelerator.gather(loss.repeat(batch_size)))

                losses = torch.cat(losses)
                losses = losses[: len(eval_dataset)]

                try:
                    perplexity = math.exp(torch.mean(losses).item())
                    perplexity = round(perplexity, 2)
                except OverflowError:
                    perplexity = float("inf")

                accuracy = round((n_correct*100/n_total).item(), 2)
                et = as_minutes(time.time()-start_time)
                accelerator.print(
                    f">>> Epoch {epoch+1} | Total Step {current_iteration} | Time: {et}"
                )
                accelerator.print(f">>> Epoch {epoch+1}: Perplexity: {perplexity}")
                accelerator.print(f">>> Epoch {epoch+1}: Accuracy: {accuracy}")

                # Save and upload ---
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(cfg.model_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(cfg.model_dir)
                torch.cuda.empty_cache()
                model.train()
                print_line()

    # --- save model at the end
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(cfg.model_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(cfg.model_dir)
    torch.cuda.empty_cache()
    model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    # execution
    main(cfg)
