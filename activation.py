from enum import Enum, member
import numpy as np


def linear(x):
    return x

def inverse(x):
    return -x

def absolute_value(x):
    return abs(x)

def unsigned_step(x):
    return 1.0 * (x > 0.0)

def sin(x):
    return np.sin(np.pi * x)

def cosine(x):
    return np.cos(np.pi * x)

def gaussian(x):
    return np.exp(-np.multiply(x, x) / 2.0)

def hyperbolic_tangent(x):
    return np.tanh(x)

def sigmoid(x):
    return (np.tanh(x / 2.0) + 1.0) / 2.0

def stanley_sigm(x):
    return 1 / (1 + np.exp(-4.9 * x))

def relu(x):
    return np.maximum(0, x)


class ActivationF(Enum):
    LINEAR = member(linear)
    INVERSE = member(inverse)
    ABS = member(absolute_value)
    UNSIGNED_STEP = member(unsigned_step)
    SIN = member(sin)
    COS = member(cosine)
    GAUSSIAN = member(gaussian)
    TANH = member(hyperbolic_tangent)
    SIGMOID = member(sigmoid)
    SSIGM = member(stanley_sigm)
    RELU = member(relu)
    
    
