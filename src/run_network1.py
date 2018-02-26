import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

new_training = [[x,y[0:4]] for [x,y] in training_data if y[0] == 1 or y[1] == 1 or y[2] == 1 or y[3] == 1]
new_test = [s for s in test_data if s[1] == 0 or s[1] == 1 or s[1] == 2 or s[1] == 3]
new_validation = [s for s in validation_data if s[1] == 0 or s[1] == 1 or s[1] == 2 or s[1] == 3]

import nework1
print("Starting")
net = nework1.Network([784, 30, 4])
net.SGD(new_training, 30, 10, 1.0, test_data=new_validation)  #epochs, mini-batch, learning rate #0.5