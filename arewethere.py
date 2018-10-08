from matplotlib import pyplot as plt
import numpy as np
import random
import math

#Opening dataset
fileopen = open('traindata') 
data_X, data_Y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)

#Parameters for neural network
hidden = 6#No of Hidden neurons!
output = 1 #Output neurons
input = 1 #Input neurons
eta = 0.01 #Learning rate
errors = []
bias = -1 #Bias
error = 0
epochs = 500 #Number of epochs

#Initialising weights
W1 = [[random.uniform(-0.2, 0.2) for i in range(input)] for i in range(hidden)]
W2 = [[random.uniform(-0.2, 0.2) for i in range(hidden)] for i in range(output)]

#Hidden neuron initialising
h_Hidden = []
for i in range(0,hidden):
    h_Hidden.append(0)
sum_Hidden = [0 for i in range(hidden)] #Will also contain sigmoid output
delta_Hidden = [0 for i in range(hidden)]
sumPrime_Hidden = [0 for i in range(hidden)]
sum_delta_weight = [0 for i in range(hidden)] #That sum thing for delta J
sigmoidHiddenSi = [0 for i in range(hidden)]

#Output neuron initialising
yhat_Output = [] #Will contain Sigmoid output
for i in range(0, output):
    yhat_Output.append(0)

sum_Output = [0 for i in range(output)]
delta_Output = [0 for i in range(output)]
sumPrime_Output = [0 for i in range(output)]

#Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x) * sigmoid((1-x))

#Forward Pass
def forward(X):
    #Input to Hidden
    for j in range(0, hidden):
        h_Hidden[j]=bias
        for k in range(0, input):
            h_Hidden[j]= h_Hidden[j]+W1[j][k]*X #this is S_j
        sum_Hidden[j] = sigmoid(h_Hidden[j]) 

    #Hidden to output
    for i in range(0, output):
        sum_Output[i]=bias
        for j in range(0, hidden):
            yhat_Output[i] = W2[i][j]*sum_Hidden[j]
            sum_Output[i] += yhat_Output[i]
        sum_Output[i] = sigmoid(sum_Output[i])  # THIS IS yHAT
    return sum_Output[i]

def backward(X, Y):
    global error
    error = Y - sum_Output[0] + error
    for i in range(0, output):
        sumPrime_Output[i] = sigmoidPrime(sum_Output[i]) #Changed from sigmoidHiddenSi to sumOutput
        delta_Output[i] = error*sumPrime_Output[i]

    for j in range(0, hidden):
        sumPrime_Hidden[j] = sigmoidPrime(h_Hidden[j]) 
        for i in range(0, output):
            sum_delta_weight[j] = delta_Output[i]*W2[i][j]

        delta_Hidden[j] = sum_delta_weight[j]*sumPrime_Hidden[j]

    #Weight updation
    for i in range(0, output):
        for j in range(0, hidden):
            W2[i][j] = W2[i][j] +eta*delta_Output[i]*sum_Hidden[j]


    for j in range(0, hidden):
        for k in range(0, input):
            W1[j][k] = W1[j][k] +eta*delta_Hidden[j]*X


#Training!
for j in range(0,epochs): #Number of epcochs
    for i in range(0,52):
        forward(data_X[i])
        backward(data_X[i],data_Y[i])
    
    meanSquare = error**2    
    errors.append(meanSquare)
    error = 0
    # print("End line")

#Plotting my errors v epochs
print("Mean Square", meanSquare)
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

#TESTING
fileopenTest = open('testing') 
data_X_test, data_Y_test = np.loadtxt(fileopenTest,usecols=(0,1), unpack=True)

diff = 0
for i in range(0,50):
    op = forward(data_X_test[i])
    diff = (data_Y_test[i]-forward(data_X_test[i]))**2+diff
print("Error OUT", diff)