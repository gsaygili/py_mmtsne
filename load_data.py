import torchvision.datasets as datasets
from pathlib import Path
import numpy as np
import os
import scipy.io as sio

def load_word_assoc_data(filename):
    print("Loading and pre-processing the word association data...")
    data = sio.loadmat(filename)
    P = data['P']
    words = [str(w[0]) for w in data['words'].flatten()]
    
    # Normalize and symmetrize P
    P /= np.sum(P, axis=1, keepdims=True)
    P = P + P.T
    P = np.maximum(P / np.sum(P), np.finfo(float).eps)
    
    print("Dataset contains:", P.shape[0], "words")
    return P, words
    
def create_mnist_subset(data, labels, size=5000):
    np.random.seed(42)
    ind = np.random.randint(0, data.shape[0], size=size)
    subdata = data[ind]
    sublabels = labels[ind]
    subdata = subdata.reshape((subdata.shape[0], subdata.shape[1]*subdata.shape[2]))
    return subdata, sublabels

