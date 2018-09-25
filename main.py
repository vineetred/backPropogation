# %matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

# each point is length, width, type (0, 1)

data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5,1]

#NEURAL NET
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.linspace(-10,10,10)
Y = sigmoid(X)
plt.plot(X,Y)
plt.show()