import argparse
import json
import os
import random
import re
import string

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import \
    DebertaV2OnlyMLMHead


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return 'e_' + ''.join(random.choice(chars) for _ in range(8))


def get_inputs(prompt, tokenizer, max_length=1296):
    return tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def mask_essay(essay, mask_token='[MASK]'):
    mask_prob = 0.1 + random.random() * 0.2
    # Split the essay into paragraphs
    paragraphs = essay.split('\n')

    masked_essay = []
    for paragraph in paragraphs:

        # Split the paragraph into sentences, keeping punctuation
        sentences = re.split(r'(?<=[.!?]) +', paragraph)

        masked_paragraph = []
        for sentence in sentences:
            mask_this_sentence = random.random() > mask_prob
            # Tokenize the sentence into words and punctuation
            tokens = re.findall(r'\b\w+\b|[.!?]', sentence)

            # Process each token, skipping the first word of the sentence
            masked_sentence = [tokens[0]] if tokens else []
            skip_next = False

            for token in tokens[1:]:
                if skip_next:
                    masked_sentence.append(token)
                    skip_next = False
                    continue
                if token in ".!?":
                    skip_next = True  # Next token is the start of a new sentence
                    masked_sentence.append(token)
                elif not mask_this_sentence:
                    if random.random() > 0.5:
                        masked_sentence.append(mask_token)
                    else:
                        masked_sentence.append(token)
                else:
                    masked_sentence.append(token)

            # Reconstruct the sentence
            masked_paragraph.append(' '.join(masked_sentence))

        # Reconstruct the paragraph
        masked_essay.append(' '.join(masked_paragraph))

    # Reconstruct the essay with paragraphs
    return '\n'.join(masked_essay)


def generate(cfg):
    accelerator = Accelerator()
    df = pd.read_csv(cfg.input_path)
    accelerator.print(f"shape of df: {df.shape}")

    df = df[df['generated'] == 0].copy()  # filter LLM essays
    df = df.reset_index(drop=True)
    accelerator.print(f"shape of df after keeping only human essays: {df.shape}")

    # df = df[~df['text'].apply(lambda x: "Ġ" in x)].copy()
    df = df.drop_duplicates(subset=['text']).copy()
    df = df.reset_index(drop=True)
    # accelerator.print(f"shape of df after removing bad examples: {df.shape}")

    df['masked_text'] = df['text'].apply(mask_essay)

    # Initialize tokenizer and model
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_model_name)
    base_config = AutoConfig.from_pretrained(cfg.backbone_model_name)

    model = AutoModelForMaskedLM.from_pretrained(cfg.backbone_model_name, config=base_config)
    model.cls = DebertaV2OnlyMLMHead(base_config)
    state_dict = dict()
    with safe_open(cfg.ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    model.load_state_dict(state_dict, strict=False)
    # set bias as zero
    model.cls.predictions.decoder.bias = nn.Parameter(torch.zeros(model.config.vocab_size))

    model = accelerator.prepare(model)
    model.eval()

    # params --
    n_examples = cfg.n_examples
    output_dir = cfg.output_dir
    progress_bar = tqdm(range(n_examples))

    #
    for i in range(n_examples):
        try:
            ex = df.sample().to_dict(orient='records')[0]
            this_example = dict()
            this_id = generate_random_string()
            this_example['id'] = this_id
            this_example['text'] = ex['text']
            this_example['masked_text'] = ex['masked_text']

            inputs = get_inputs(ex['masked_text'], tokenizer, max_length=cfg.max_length)
            device = accelerator.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            mask_token_indices = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1].tolist()

            with torch.no_grad():
                output = model(**inputs)

            # Sample from the top 10 tokens for each mask
            top_k = cfg.top_k
            sampled_token_ids = []

            for mask_index in mask_token_indices:
                predicted_token_ids = output[0].argsort(dim=-1, descending=True)[0, mask_index, :top_k]
                sampled_token_id = random.choice(predicted_token_ids).item()
                sampled_token_ids.append(sampled_token_id)

            # Replace mask tokens with sampled tokens in the input_ids
            for mask_index, token_id in zip(mask_token_indices, sampled_token_ids):
                inputs['input_ids'][0, mask_index] = token_id

            final_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

            this_example['demasked_text'] = final_text

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
