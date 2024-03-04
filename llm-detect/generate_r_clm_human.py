import argparse
import json
import os
import random
import string

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return 'e_' + ''.join(random.choice(chars) for _ in range(8))


def get_instruction(inputs):
    n_chars = random.randint(64, 128)
    ret = f"""
Prompt: {inputs['prompt_name']}
Start: {inputs['text'][:n_chars]}
    """.strip()
    ret = f"### Instruction:\n{ret}\n\n### Response:"

    return ret


def get_inputs(prompt, tokenizer, n=1):
    return tokenizer([prompt]*n, return_tensors="pt")


def process_response(texts):
    ret = []

    for text in texts:
        text = text.split("### Response:")[-1].split("</s>")[0].strip()
        text = text.replace("<unk>", "")
        ret.append(text)
    return ret


def pre_process_essay(essay_df):
    essay_df["instruction"] = essay_df.apply(get_instruction, axis=1)
    essay_df["prompt"] = essay_df['instruction']
    # essay_df["prompt"] = essay_df['prompt_name']

    return essay_df


def generate(cfg):
    accelerator = Accelerator()

    essay_df = pd.read_csv(cfg.input_data_path)
    essay_df = pre_process_essay(essay_df)

    prompts = essay_df["prompt"].unique().tolist()
    print(f"Number of prompts: {len(prompts)}")

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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        low_cpu_mem_usage=True,
        # quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True

    )

    model = PeftModel.from_pretrained(base_model, cfg.adapter_path)
    model = model.merge_and_unload()
    model = accelerator.prepare(model)
    model.eval()

    n_examples = cfg.n_examples
    n_gen_per_prompt = cfg.n_gen_per_prompt
    output_dir = cfg.output_dir

    progress_bar = tqdm(range(n_examples))

    for i in range(n_examples):
        # print(f"---- Example {i+1}/{n_examples} ------")
        temperature = 0.4 + 1.0 * random.random()
        top_k = random.randint(4, 128)
        penalty_alpha = 0.4 + 0.2 * random.random()
        guidance_scale = 1.0 + 0.25 * random.random()
        # typical_p = 0.8 + 0.2 * random.random()

        generation_config = GenerationConfig.from_pretrained(
            cfg.base_model_path,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            penalty_alpha=penalty_alpha,
            guidance_scale=guidance_scale,
            # typical_p=typical_p,
            max_new_tokens=cfg.max_num_tokens,
            pad_token_id=tokenizer.pad_token_id,

            # repetition_penalty=1.4,
            # top_p=0.95,
        )

        try:
            prompt = random.choice(prompts)
            # start = essay_df[essay_df['prompt'] == prompt].sample(1)['instruction'].values[0]
            start = prompt

            this_example = dict()
            this_id = generate_random_string()
            this_example['id'] = this_id
            this_example['prompt'] = prompt
            this_example['start'] = start
            this_example['temperature'] = temperature
            this_example['top_k'] = top_k
            this_example['guidance_scale'] = guidance_scale
            this_example['penalty_alpha'] = penalty_alpha
            # this_example['typical_p'] = typical_p

            inputs = get_inputs(prompt, tokenizer, n=n_gen_per_prompt)
            inputs = get_inputs(start, tokenizer, n=n_gen_per_prompt)

            device = accelerator.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # print(f"input device = {inputs['input_ids'].device}")
            # print(f"model device = {model.device}")
            # print("--"*40)
            inputs["use_cache"] = False  # for Phi-2
            with torch.no_grad():
                output = model.generate(**inputs, generation_config=generation_config)
            output = tokenizer.batch_decode(output)

            output = process_response(output)
            this_example['responses'] = output

            with open(f"{output_dir}/{this_id}.json", "w") as f:
                json.dump(this_example, f)

        except Exception as e:
            print(e)
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
