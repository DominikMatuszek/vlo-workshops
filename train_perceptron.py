import json
import random

# Load the train.json file
data = json.load(open('train.json'))

num_params = len(data[0]) - 1
num_examples = len(data)

weights = [0] * num_params

num_steps = 100000

for epoch in range(num_steps):
    example = random.choice(data)

    label = example[-1]
    features = example[:-1]

    weighted_sum = 0

    for i in range(num_params):
        weighted_sum += weights[i] * features[i]
    
    prediction = 1 if weighted_sum > 0 else -1

    learning_rate = 0.1

    if prediction != label:
        for i in range(num_params):
            weights[i] += learning_rate * label * features[i]
        

print("Weights: %s" % weights)
