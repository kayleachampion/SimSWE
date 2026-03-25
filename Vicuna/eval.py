import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm

MODEL_PATH = "./vicuna-tinyllama"
DATA_PATH = "./vicuna_test.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

data = json.load(open(DATA_PATH))

scores = []

for item in tqdm(data[:100]):   # limit for speed
    instruction = item["instruction"]
    inp = item["input"]
    target = item["output"]

    if inp.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Output:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Output:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    pred = text.split("### Output:")[-1].strip()

    score = scorer.score(target, pred)["rougeL"].fmeasure
    scores.append(score)

print("Avg ROUGE-L:", sum(scores) / len(scores))
print("The purpose here is to determine: does the model produce outputs that are similar to the training data? The higher the better -- perhaps .7.")
