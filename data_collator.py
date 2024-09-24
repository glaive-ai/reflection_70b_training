import warnings
from typing import List, Union, Any, Dict
import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling

class CustomDualMaskCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        response_template: Union[str, List[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(tokenizer, *args, mlm=mlm, **kwargs)
        hardcoded_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.response_template = self.tokenize_template(response_template)
        self.hardcoded_template = self.tokenize_template(hardcoded_template)
        self.ignore_index = ignore_index

    def tokenize_template(self, template):
        if isinstance(template, str):
            return self.tokenizer.encode(template, add_special_tokens=False)
        return template

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        new_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for i in range(len(examples)):
            response_start = self.find_template_start(batch["input_ids"][i], self.response_template)
            hardcoded_start = self.find_template_start(batch["input_ids"][i], self.hardcoded_template)

            if response_start is not None:
                # Create two samples
                new_batch["input_ids"].extend([batch["input_ids"][i], batch["input_ids"][i]])
                new_batch["attention_mask"].extend([batch["attention_mask"][i], batch["attention_mask"][i]])
                
                labels1 = batch["input_ids"][i].clone()
                labels1[:response_start] = self.ignore_index
                
                labels2 = batch["input_ids"][i].clone()
                labels2[:hardcoded_start] = self.ignore_index
                
                new_batch["labels"].extend([labels1, labels2])
            else:
                # Create one sample
                new_batch["input_ids"].append(batch["input_ids"][i])
                new_batch["attention_mask"].append(batch["attention_mask"][i])
                
                labels = batch["input_ids"][i].clone()
                labels[:hardcoded_start] = self.ignore_index
                
                new_batch["labels"].append(labels)

        # Convert lists to tensors
        for key in new_batch:
            new_batch[key] = torch.stack(new_batch[key])

        return new_batch

    def find_template_start(self, input_ids, template):
        input_ids = input_ids.tolist()
        for i in range(len(input_ids) - len(template) + 1):
            if input_ids[i:i+len(template)] == template:
                return i + len(template)
        return None