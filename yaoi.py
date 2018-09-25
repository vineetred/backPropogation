from matplotlib import pyplot as plt
import numpy as np
import random
data_X = []

#Opening the file
fileopen = open('traindata')
data = np.loadtxt(fileopen, usecols=(0))

#Parameters for neural network
hidden = 3
output = 1
input = 1
eta = 0.5

#Random weights
W1 = np.random.randn(input, hidden)
W2 = np.random.randn(hidden, output)
sumHidden_J = [[0 for i in range(input)] for j in range(hidden)]
sumOutput_J = [[0 for i in range(hidden)] for j in range(output)]
deltaHidden = [[0 for i in range(input)] for j in range(hidden)]
fPrimeOutput = [[0 for i in range(hidden)] for j in range(output)]
fPrimeHidden = [[0 for i in range(input)] for j in range(hidden)]
deltaOutput = [[0 for i in range(hidden)] for j in range(output)]
# print(sumHidden_J)
# print(W1)
error = 0
# print(W1[0][1])
#Sgimoud
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * (1-sigmoid(x))

def forward(X):
    #INPUT --> HIDDEN
    for i in range(0, hidden):
        for j in range(0, input):
            sumHidden_J[i] = sumHidden_J[i]+ W1[j][i]*X
            sumHidden_J[i] = sigmoid(sumHidden_J[i])

    #HIDDEN --> OUTPUT
    for i in range(0,output):
        for j in range(0,hidden):
            sumOutput_J[i] = sumOutput_J[i] + W2[j][i]*sumHidden_J[j]
            sumOutput_J[i] = sigmoid(sumOutput_J[i])

def backward(X, Y):

    error = 0.5*((Y - sumOutput_J[0])**2)
    print(error)
    # fPrimeOutput = sigmoidPrime(sumOutput_J) 
    # deltaOutput = error*fPrimeOutput

    # fPrimeHidden = sigmoidPrime(sumHidden_J)
    for i in range(0,output):
        for j in range(0, hidden):
            fPrimeOutput[i] = sigmoidPrime(sumOutput_J[i])
            deltaOutput[i] = error*fPrimeOutput[i] 
            fPrimeHidden[j] = sigmoidPrime(sumHidden_J[j])
            deltaHidden[j] = deltaOutput[i]*W2[j][i]*fPrimeHidden[j]


    #Weight UPDATIONS
    #Hidden --> Output
    for i in range(0,output):
        for j in range(0, hidden):
            W2[j][i] += (eta*deltaOutput[i][j]*sumHidden_J[j])

    #Input --> Hidden
    for i in range(0,hidden):
        for j in range(0,input):
            W1[j][i] += eta*deltaHidden[i][j]*X

for i in range(0,1000):
    forward(2)
    backward(2,3)

# print(sumHidden_J)
# print(sumOutput_J)
# print(deltaHidden)
