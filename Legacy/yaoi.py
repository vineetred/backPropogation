from matplotlib import pyplot as plt
import numpy as np
import random
# data_X = []

#Opening the file
fileopen = open('traindata')

# data_X = np.loadtxt(fileopen, usecols=(1))
# data_Y = np.loadtxt(fileopen, usecols=(0))

data_X = [15,16,17,18,19,20,21,22,23,24,25]
data_Y = [13,14,15,16,17,18,19,20,21,22,23]

#Parameters for neural network
hidden = 3
output = 1
input = 1
eta = 0.1
errors = []

#Random weights
W1 = np.random.randn(input, hidden)
W2 = np.random.randn(hidden, output)
sumHidden_J = [[0 for i in range(input)] for j in range(hidden)]
sumOutput_J = [[0 for i in range(hidden)] for j in range(output)]
# deltaHidden = [[0 for i in range(input)] for j in range(hidden)]
deltaHidden = []

for i in range(0, hidden):
    deltaHidden.append(0)



fPrimeOutput = [[0 for i in range(hidden)] for j in range(output)]
fPrimeHidden = [[0 for i in range(input)] for j in range(hidden)]
deltaOutput = [[0 for i in range(hidden)] for j in range(output)]
sumDeltaHidden = []
for i in range(0, hidden):
    sumDeltaHidden.append(0)
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

    error = Y - sumOutput_J[0]
    # print(sumOutput_J[0])
    print(error)
    errors.append(error)
    # fPrimeOutput = sigmoidPrime(sumOutput_J) 
    # deltaOutput = error*fPrimeOutput

    # fPrimeHidden = sigmoidPrime(sumHidden_J)
    for i in range(0,output):
        fPrimeOutput[i] = sigmoidPrime(sumOutput_J[i])
        deltaOutput[i] = error*fPrimeOutput[i] 


    for j in range(0, hidden):
        fPrimeHidden[j] = sigmoidPrime(sumHidden_J[j])
        for i in range (0, output):
            sumDeltaHidden[j] += deltaOutput[i]*W2[j][i]
        deltaHidden[j]= sumDeltaHidden[j]*fPrimeHidden[j]


    # for i in range(0,hidden):
    #     for j in range(0, input):
    #         fPrimeHidden[j] = sigmoidPrime(sumHidden_J[j])
    #         deltaHidden[j] = deltaOutput[i]*W2[j][i]*fPrimeHidden[j]


    #Weight UPDATIONS
    #Hidden --> Output
    for i in range(0,output):
        for j in range(0, hidden):
            W2[j][i] += (eta*deltaOutput[i][j]*sumHidden_J[j])

    #Input --> Hidden
    for i in range(0,hidden):
        for j in range(0,input):
            W1[j][i] += eta*deltaHidden[i]*X
    return sumOutput_J[0]

for j in range(0,10):
    for i in range(0,10):
        forward(data_X[i])
        backward(data_X[i],data_Y[i])
        
        # print(hello)
    print("End line")
plt.plot(errors)
# print(data_X)
# print(data_Y)
plt.show()
# print(sumHidden_J)
# print(sumOutput_J)
# print(deltaHidden)
