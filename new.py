import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], 
            [1, 5],
            [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
print(X)
# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100



#Neuron class
class Neural_Network(object):
    def __init__(self):

        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        # self.z = (self.W1*X)
        print (self.z)
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        print ("O",o)
        return o 

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print ("Input: \n" + str(X)) 
  print ("Actual Output: \n" + str(y)) 
  print ("Predicted Output: \n" + str(NN.forward(X))) 
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")
  NN.train(X, y)