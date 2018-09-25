from matplotlib import pyplot as plt
import numpy as np
data_X = []

#Opening the file
fileopen = open('traindata')
data = np.loadtxt(fileopen, usecols=(0))
# data_X = data[:,0]
# print(data)
# print(data[0])
#Parameters for neural network
hidden = 3
output = 1
input = 2
eta = 0.1

#Weight activation
W1 = np.random.randn(input, hidden) #Input to Hidden
W2 = np.random.randn(hidden, output) #Hidden to output
bias = np.random.randn()
# print(W2.T[0])
# print (len(W1.T))

#Sigmoid functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * (1-sigmoid(x))

def forward(X):
    forward.sum = np.dot(X, W1) #Input ----> Hidden weight
    # print ("Sum", sum)
    forward.sumSig = sigmoid(forward.sum)
    # print("Sumsig", sumSig)
    forward.sum2 = np.dot(forward.sumSig, W2)
    # print("sum2", sum2) #Hidden ----> Output
    forward.sum2Sig = sigmoid(forward.sum2)
    return forward.sum2Sig

# def forward2(X):
#     for i in range(0, len(W1.T)):
#         for j in range(0, len(W1)):
#             sum[j][k] = X*W1[j][k]
#             sumSig[j][k] = sigmoid(sum[j][k])

#     for i in range(0, len(W2)):
#         sum2[i] = sumSig[]


def backwardProp(X, Y):
    error = forward.sum2Sig - Y
    delta = error*sigmoidPrime(forward.sum2Sig)
    print(delta)
    sum2_error = forward.sumSig 


    for j in range(0, len(W1.T)):
        for k in range(0,len(W1)):
            W1[j][k] = W1[j][k] + eta* error * X

    for i in range(0,len(W2)):
        W2[i] = W2[i] + eta * delta[i] * forward.sum2Sig[i]
    
    
    
hello = forward(10)
world = backwardProp(10, 9) 
# print(hello)