import pickle
import gzip
import numpy as np

def vectorized_result(j: int):
    """Vectorize scalar j into a one-hot vector"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data(filepath='./data/mnist.pkl.gz'):
    with gzip.open(filepath, 'rb') as f:
        # See at https://github.com/MichalDanielDobrzanski/DeepLearningPython/issues/15
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
        return (training_data, validation_data, test_data)
    
def load_data_wrapper():
    """Specific to MNIST"""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)