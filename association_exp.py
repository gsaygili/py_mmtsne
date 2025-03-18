#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:27:22 2025

@author: genai
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def load_data(filename):
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

def run_tsne(P):
    no_dims = 2
    no_maps = 8 if P.shape[0] == 1000 else 25
    max_iter = 500
    
    print(f"Running multiple maps t-SNE to construct {no_maps} maps of {P.shape[0]} words...")
    if P.shape[0] == 5000:
        print("This may take up to 24 hours to compute!")
    else:
        print("This may take up to 15 minutes to compute!")
    
    from mult_maps_tsne import mult_maps_tsne  # Assuming the function is implemented
    maps, weights = mult_maps_tsne(P, no_maps, no_dims, max_iter)
    return maps, weights

def plot_maps(maps, weights, words):
    print("Drawing maps...")
    
    for m in range(maps.shape[2]):
        plt.figure(figsize=(8, 6))
        
        # Filter words with sufficient importance weight
        indices = np.where(weights[:, m] > 0.05)[0] if maps.shape[2] > 1 else np.arange(maps.shape[0])
        
        # Scatter plot
        plt.scatter(maps[indices, 0, m], maps[indices, 1, m], s=weights[indices, m] * 40, alpha=0.6)
        
        # Add text labels
        x_min, x_max = plt.xlim()
        width = x_max - x_min
        for i in indices:
            plt.text(maps[i, 0, m] + 0.006 * width, maps[i, 1, m], words[i], fontsize=8)
        
        plt.axis('off')
        plt.title(f"t-SNE Map {m+1}")
        plt.show()

if __name__ == "__main__":
    filename = 'association1000.mat'  # Update with actual file path
    P, words = load_data(filename)
    maps, weights = run_tsne(P)
    plot_maps(maps, weights, words)
