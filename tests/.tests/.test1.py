import numpy as np
import tensorflow as tf

a = np.arange(100)

def generate_training_data(data:np.array):
    while True:
        a =  np.arange(data.shape[1])
        b = a.reshape(a.shape[0], 1)
        yield b, np.array([1,])