import json
import random

data = json.load(open("vicuna_generated_data.json"))

random.shuffle(data)

split = int(len(data) * 0.9)

train = data[:split]
test = data[split:]

json.dump(train, open("vicuna_train.json", "w"), indent=2)
json.dump(test, open("vicuna_test.json", "w"), indent=2)

print("train:", len(train))
print("test:", len(test))
