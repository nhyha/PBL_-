import argparse
import ast
import json
import os
import random
import string
import traceback
from copy import deepcopy

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return 'e_' + ''.join(random.choice(chars) for _ in range(8))


def get_instruction(inputs):
    ret = f"""
Prefix: {inputs['prefix']}
Suffix: {inputs['suffix']}
    """.strip()

    ret = f"### Instruction:\n{ret}\n\n### Response:"
    return ret


def get_inputs(prompts, tokenizer):
    return tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=1296,
        return_tensors="pt"
    )


def process_response(texts):
    ret = []

    for text in texts:
        text = text.split("### Response:")[-1].split("</s>")[0].strip()
        text = text.replace("<unk>", "")
        ret.append(text)
    return ret


def get_inference_dataset(essay_df):
    essay_df = deepcopy(essay_df)
    ids = []
    sentence_ids = []
    prefix = []
    suffix = []
    target = []

    MAX_SENTS = 16

    for idx, row in essay_df.iterrows():
        essay_id = row['id']
        elements = row['processed_discourse_text'][:MAX_SENTS]

        for idx in range(len(elements)):
            ids.append(essay_id)
            sentence_ids.append(idx)
            target.append(elements[idx])
            prefix.append(" ".join(elements[:idx]))
            suffix.append(" ".join(elements[idx+1:]))

    infer_df = pd.DataFrame()
    infer_df['id'] = ids
    infer_df['sentence_id'] = sentence_ids
    infer_df['prefix'] = prefix
    infer_df['suffix'] = suffix
    infer_df['target'] = target

    infer_df['prefix'] = infer_df['prefix'].fillna("")
    infer_df['suffix'] = infer_df['suffix'].fillna("")
    infer_df["prompt"] = infer_df.apply(get_instruction, axis=1)
    return infer_df


def generate(cfg):
    accelerator = Accelerator()

    essay_df = pd.read_csv(cfg.input_data_path)

    with open(cfg.mapping_data_path) as f:
        prompt_map = json.load(f)

    essay_df['prompt_name'] = essay_df['id'].apply(lambda x: prompt_map.get(x, "NA"))

    keep_prompts = [
        # '"A Cowboy Who Rode the Waves"',
        # "The Face on Mars",

        # "Facial action coding system",
        "Driverless cars",
        "Exploring Venus",

        "Does the electoral college work?",
        "Car-free cities",
    ]

    essay_df = essay_df[essay_df['prompt_name'].isin(keep_prompts)].copy()
    essay_df = essay_df.reset_index(drop=True)

    essay_df['processed_discourse_text'] = essay_df['processed_discourse_text'].apply(ast.literal_eval)
    infer_df = get_inference_dataset(essay_df)
    all_essay_ids = infer_df['id'].unique().tolist()

    # infer_df['prompt_name'] = infer_df['id'].apply(lambda x: prompt_map.get(x, "NA"))
    # accelerator.print(infer_df.prompt_name.value_counts())
    # keep_prompts = [
    #     '"A Cowboy Who Rode the Waves"',
    #     "The Face on Mars",
    # ]
    # infer_df = infer_df[infer_df['prompt_name'].isin(keep_prompts)].copy()
    # infer_df = infer_df.reset_index(drop=True)
    # accelerator.print(f"Number of examples: {len(infer_df)}")
    # accelerator.print(infer_df.head(10))

    # model & tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_path,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, cfg.adapter_path)
    model = model.merge_and_unload()
    model = accelerator.prepare(model)
    model.eval()

    n_examples = cfg.n_examples
    output_dir = cfg.output_dir

    progress_bar = tqdm(range(n_examples))

    for i in range(n_examples):
        # print(f"---- Example {i+1}/{n_examples} ------")
        temperature = 0.25 + 1.25 * random.random()
        top_k = random.randint(16, 256)
        penalty_alpha = 0.1 + 0.8 * random.random()
        guidance_scale = 1.0 + 0.3 * random.random()
        eta_cutoff = 3e-4 + 1e-3 * random.random()

        generation_config = GenerationConfig.from_pretrained(
            cfg.base_model_path,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            penalty_alpha=penalty_alpha,
            guidance_scale=guidance_scale,
            max_new_tokens=cfg.max_num_tokens,
            pad_token_id=tokenizer.pad_token_id,
            # renormalize_logits=True,
            eta_cutoff=eta_cutoff,
        )

        try:
            essay_id = random.choice(all_essay_ids)
            focus_df = infer_df[infer_df['id'] == essay_id].copy()
            prompts = focus_df['prompt'].values.tolist()

            this_example = dict()
            this_id = generate_random_string()
            this_example['id'] = this_id
            this_example['base_id'] = essay_id
            this_example['prompts'] = prompts
            this_example['temperature'] = temperature
            this_example['top_k'] = top_k
            this_example['guidance_scale'] = guidance_scale
            this_example['penalty_alpha'] = penalty_alpha

            inputs = get_inputs(prompts, tokenizer)
            device = accelerator.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, generation_config=generation_config)
            output = tokenizer.batch_decode(output)

            output = process_response(output)
            this_example['responses'] = output

            with open(f"{output_dir}/{this_id}.json", "w") as f:
                json.dump(this_example, f)

        except Exception as e:
            # print(e)
            traceback.print_exc()
        progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # execution
    generate(cfg)
