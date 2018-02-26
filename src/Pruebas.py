import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

new_training = []
for sample in training_data:
    new_sample = []
    output = [[1.0],[0.0]]
    if sum(sample[1][5:10]) == 1:
        output = [[0.0],[1.0]]
    new_sample.append(sample[0])
    new_sample.append(output)
    new_sample = np.array(new_sample)
    new_training.append(new_sample)

new_validation = []
for sample in validation_data:
    new_sample = []
    output = 0
    if sample[1] > 4 == 1:
        output = 1
    new_sample.append(sample[0])
    new_sample.append(output)
    new_validation.append(new_sample)

new_test = []
for sample in test_data:
    new_sample = []
    output = 0
    if sample[1] > 4 == 1:
        output = 1
    new_sample.append(sample[0])
    new_sample.append(output)
    new_test.append(new_sample)

import network2
print("Starting...")
net = network2.Network([784, 30, 2], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(new_training, 30, 10, 0.1, lmbda = 5.0, evaluation_data=new_validation, monitor_evaluation_accuracy=True) #5.0
