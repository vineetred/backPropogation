from matplotlib import pyplot as plt
import numpy as np
import random
import math

# data_X = []

#FORMAT = <whatparameter><whichneuron>

#Opening the file
fileopen = open('traindata')

# data_X = np.loadtxt(fileopen, usecols=(1))
# data_Y = np.loadtxt(fileopen, usecols=(0))

data_X = [15,16,17,18,19]
data_Y = [13,14,15,16,17]

#Parameters for neural network
hidden = 3
output = 1
input = 1
eta = 0.01
errors = []
bias = 1

# W1 = np.random.randn(input, hidden)

W1 = [[random.uniform(-1.0, 1.0) for i in range(input)] for i in range(hidden)]
W2 = [[random.uniform(-1.0, 1.0) for i in range(hidden)] for i in range(output)]
# print (W1)
# print (W2)
# 
#Hidden
h_Hidden = []
for i in range(0,hidden):
    h_Hidden.append(0)
sum_Hidden = [0 for i in range(hidden)] #Will also contain sigmoid output
delta_Hidden = [0 for i in range(hidden)]
sumPrime_Hidden = [0 for i in range(hidden)]
sum_delta_weight = [0 for i in range(hidden)] #That sum thing for delta J
sigmoidHiddenSi = [0 for i in range(hidden)]

#Output
yhat_Output = [] #Will contain Sigmoid output
for i in range(0, output):
    yhat_Output.append(0)

sum_Output = [0 for i in range(output)]
delta_Output = [0 for i in range(output)]
sumPrime_Output = [0 for i in range(output)]

#Sigmoidssss
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    return x * (1-x)

def forward(X):
    #Input to Hidden
    for j in range(0, hidden):
        h_Hidden[j] = 0
        for k in range(0, input):
            h_Hidden[j]= h_Hidden[j]+W1[j][k]*X +bias
        sum_Hidden[j] = sigmoid(h_Hidden[j])

    #Hidden to output
    for i in range(0, output):
        for j in range(0, hidden):
            sum_Output[i]=0
            yhat_Output[i] = W2[i][j]*sum_Hidden[j] + bias
            # print("yhat ",yhat_Output[i])
            sum_Output[i] += yhat_Output[i]
        sigmoidHiddenSi[i] = sum_Output[i]
        # print("S_I", sigmoidHiddenSi)
        sum_Output[i] = sigmoid(sum_Output[i])
        # print("Sum_OUTPUT", sum_Output)


def backward(X, Y):
    error = Y - sum_Output[0]
    for i in range(0, output):
        sumPrime_Output[i] = sigmoidPrime(sigmoidHiddenSi[i])
        # print("Sumprime 2" ,sumPrime_Output[i])
        delta_Output[i] = error*sumPrime_Output[i]
        # print("Delta Output:  ",delta_Output)

    for j in range(0, hidden):
        sumPrime_Hidden[j] = sigmoidPrime(h_Hidden[j]) 
        for i in range(0, output):
            sum_delta_weight[j] = delta_Output[i]*W2[i][j]

        delta_Hidden[j] = sum_delta_weight[j]*sumPrime_Hidden[j]
    # print("Delta Hidden ", delta_Hidden)

    #Weight updating
    for i in range(0, output):
        for j in range(0, hidden):
            # print(W2)
            W2[i][j] = W2[i][j] - eta*delta_Output[i]*sum_Hidden[j]
            # delta_Output[i]=0
            # print("Delta", delta_Output)
            # print("H_HIDDEN", h_Hidden)

    for j in range(0, hidden):
        for k in range(0, input):
            W1[j][k] = W1[j][k] - eta*delta_Hidden[j]*X
            # delta_Hidden[j]=0

    
    # errSquare = (Y -sum_Output[0]**2)
    print ("Error", error)
    # errors.append(errSquare)


# data_X = [0.1,0.2,0.3,0.4,0.5]
# data_Y = [0.2,0.3,0.4,0.5,0.6]

for j in range(0,10): #Number of epcochs
    for i in range(0,5):
        forward(data_X[i])
        backward(data_X[i],data_Y[i])
        
        # print(hello)
    print("End line")
print (W1)
print (W2)
# plt.plot(errors)
# plt.show()
# print(sigmoidPrime(10))