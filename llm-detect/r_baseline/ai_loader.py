
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class AiCollator(DataCollatorWithPadding):
    """
    data collector for LLM Detect AI Generated Text task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):

        buffer_dict = dict()
        buffer_keys = ["id"]

        for key in buffer_keys:
            if key in features[0].keys():
                value = [feature[key] for feature in features]
                buffer_dict[key] = value

        labels = None
        if "generated" in features[0].keys():
            labels = [feature["generated"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        for key, value in buffer_dict.items():
            batch[key] = value

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

        return batch


# ---
def show_batch(batch, tokenizer, n_examples=16, task='training'):

    bs = batch['input_ids'].size(0)
    print(f"batch size: {bs}")

    print(f"shape of input_ids: {batch['input_ids'].shape}")

    n_examples = min(n_examples, bs)
    print(f"Showing {n_examples} from a {task} batch...")

    print("\n\n")
    for idx in range(n_examples):
        print(f"Example {idx+1} | id: {batch['id'][idx]}")
        print(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        # print("\n\n")

        if "infer" not in task.lower():
            print("--"*20)
            labels = batch['labels'][idx]
            print(f"Label: {labels}")
        print('=='*40)
