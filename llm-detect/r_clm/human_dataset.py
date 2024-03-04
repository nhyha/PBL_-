from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100
N_CHAR_PREFIX = 128


def get_tokenizer(cfg):

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=cfg.model.tokenizer.use_fast,
        padding_side=cfg.model.tokenizer.padding_side,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# --------------- Dataset ----------------------------------------------#


def get_instruction(inputs):
    ret = f"""
Prompt: {inputs['prompt_name']}
Start: {inputs['text'][:N_CHAR_PREFIX]}
    """.strip()
    return ret


class AiDataset:
    """
    Dataset class for LLM Detect AI Generated Text competition
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)

    def format_source(self, instruction):
        ret = f"### Instruction:\n{instruction}\n\n### Response: "
        return ret

    def format_target(self, response):
        return f"{response[N_CHAR_PREFIX:]} {self.tokenizer.eos_token}"

    def tokenize_function(self, examples):
        sources = [self.format_source(s) for s in examples["instruction"]]
        targets = [self.format_target(t) for t in examples["text"]]
        chats = [s + t for s, t in zip(sources, targets)]

        ex_tokenized_inputs = self.tokenizer(
            chats,
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
        )

        src_tokenized_inputs = self.tokenizer(
            sources,
            padding=False,
            truncation=False,
        )

        src_lens = [len(s)-1 for s in src_tokenized_inputs["input_ids"]]
        input_ids = ex_tokenized_inputs["input_ids"]
        attention_mask = ex_tokenized_inputs["attention_mask"]
        labels = deepcopy(input_ids)

        for idx, src_len in enumerate(src_lens):
            labels[idx][:src_len] = [IGNORE_INDEX] * src_len

        to_return = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return to_return

    def preprocess_function(self, persuade_df):
        persuade_df["instruction"] = persuade_df.apply(get_instruction, axis=1)
        return persuade_df

    def get_dataset(self, df):
        df = deepcopy(df)
        df = self.preprocess_function(df)
        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=task_dataset.column_names
        )
        return task_dataset
