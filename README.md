# backPropogation

Using Python to build a backpropogating neural network from scratch for CS303. It uses the Numpy and Matplotlib libraries. Make sure you have installed that before using my code. 

The activation function we use is the **Sigmoid** function, hence the training and the testing data needs to be normalised.

The parameters of the neural network can be edited all at once in the following section of the code - 
```
#Parameters for neural network
hidden = 6 #No of Hidden neurons!
output = 1 #Output neurons
input = 1 #Input neurons
eta = 0.01 #Learning rate
errors = []
bias = 1 #Bias
epochs = 2500 #Number of epochs
```

The dataset that this neural network works on is normalised. If you're using it on another dataset, make sure it is normailsed.
