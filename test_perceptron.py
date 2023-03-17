import json

weights = json.load(open('weights.json'))
num_params = len(weights)

print("Weights: %s" % weights)

# Load the test.json file
test = json.load(open('test.json'))

test_cases = len(test)
successes = 0

for example in test:
    label = example[-1]

    features = example[:-1]

    weighted_sum = 0

    for i in range(num_params):
        weighted_sum += weights[i] * features[i]
    
    prediction = 1 if weighted_sum > 0 else -1


    if prediction == label:
        successes += 1

print("Successes:", successes)
print("Test cases:", test_cases)


print("Accuracy:", (successes / test_cases) * 100)