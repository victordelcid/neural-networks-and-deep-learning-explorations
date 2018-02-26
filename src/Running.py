import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#print training_data[0]

new_training = [[x,y[0:4]] for [x,y] in training_data if y[0] == 1 or y[1] == 1 or y[2] == 1 or y[3] == 1]
new_test = [s for s in test_data if s[1] == 0 or s[1] == 1 or s[1] == 2 or s[1] == 3]
new_validation = [s for s in validation_data if s[1] == 0 or s[1] == 1 or s[1] == 2 or s[1] == 3]


import network2
print("Starting")
net = network2.Network([784, 30, 4], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(new_training, 30, 10, 1.0, lmbda = 5.0, evaluation_data=new_validation, monitor_evaluation_accuracy=True) #5.0