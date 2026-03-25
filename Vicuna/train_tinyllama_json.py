import json
import torch
import re
from datasets import Dataset
from generate_instruction import encode_prompt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "vicuna_train.json"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    trust_remote_code=True,
    revision="main"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    revision="main"
)


# -------------------------
# load JSON directly
# -------------------------

data = json.load(open(DATA_PATH))

examples = []

for item in data:

    instruction = item["instruction"]
    input_text = item["input"]
    output = item["output"]

    prompt = encode_prompt([
        {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }
    ])

    examples.append({"text": prompt})


dataset = Dataset.from_list(examples)


# -------------------------
# tokenizer
# -------------------------

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized = dataset.map(tokenize)



# -------------------------
# training args
# -------------------------

args = TrainingArguments(
    output_dir="./vicuna-tinyllama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)

trainer.train()

trainer.save_model("./vicuna-tinyllama")
