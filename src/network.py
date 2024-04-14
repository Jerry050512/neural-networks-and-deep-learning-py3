import random
import numpy as np

def sigmoid(z): 
    """Sigmoid Activation Function"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): 
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    def __init__(self, sizes: list):
        """Init the network with sizes of each layer"""
        self.num_layers = len(sizes)
        self.sizes = tuple(sizes)   
        # Note that if the original sizes was modified won't affect this property
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs: int, mini_batch_size: int, 
            eta=3, test_data=None):
        """## Standard Gradient Descent
        ### Parameters
        - training_data: [(x, y)]
        - eta: learning rate
        - test_data: [(x, y)]"""
        training_data = training_data
        if test_data is not None:
            test_data = test_data
            n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print(f"Epoch {epoch:3d}: {self.evaluate(test_data):4d} / {n_test}")
            else:
                print(f"Epoch {epoch:3d} complete")

    def update_mini_batch(self, mini_batch, eta=3):
        """Update the networkd with the given mini batch by applyiing gradient 
        descent using backpropagation."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """### Parameters:
        - x: input data
        - y: target data
        - x, y are all training data.
        ### Return: 
        - a tuple `(nabla_b, nabla_w)` representing the gradient 
        for the cost function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # Note that the `int(x == y) for (x, y) in test_results` 
        # is a generator object.
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Vector of partial derivatives of the activation"""
        return output_activations - y