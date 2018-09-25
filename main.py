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

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

X = np.linspace(-100,100,100)
Y = sigmoid_p(X)
# plt.plot(X,Y)
# plt.show()


for i in range(100):
	ri = np.random.randint(len(data))
	point = data[ri]

	z = point[0]*w1 + point[1]*w2 + b
	target = point[2]
	pred = sigmoid(z)

	error = (pred - target)**2
	print (target, error)