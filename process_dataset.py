from datasets import load_dataset
import json

dataset = load_dataset("glaiveai/reflection-v1",split="train")

def format_row(row):
    return f'''<|start_header_id|>system<|end_header_id|>{row["system"]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{row["prompt"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{row["response"]}<|eot_id|>'''

processed_dataset = [{"text":format_row(row)} for row in dataset]

with open("reflection-data.json","w") as f:
    json.dump(processed_dataset,f)
