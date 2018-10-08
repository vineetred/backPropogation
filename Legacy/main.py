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

eta = 0.01
costs = []
for i in range(50000):
	ri = np.random.randint(len(data))
	point = data[ri]

	S_i = point[0]*w1 + point[1]*w2 + b
	target = point[2]
	pred = sigmoid(S_i)

	error = np.square(pred - target)

	derErrorPred = 2*(pred-target)
	derPredSi = sigmoid_p(S_i)
	derSiw1 = point[0]
	derSiw2 = point[2]
	derSibias = 1

	derErrorw1 = derErrorPred * derPredSi * derSiw1
	derErrorw2 = derErrorPred * derPredSi * derSiw2
	derErrorbias = derErrorPred * derPredSi * derSibias
	
	#Training
	w1 = w1 - eta * derErrorw1
	w2 = w2 - eta * derErrorw2
	b = b - eta * derErrorbias
	# if(i%1000==0):
	costs.append(error)

plt.plot(costs)
plt.show()