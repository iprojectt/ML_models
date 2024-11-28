import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min



#  3.1 Implementing a K-Means Class
class KMeans:
    def __init__(self, k=3, max_iters=200):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X):
        self.X = X
        n_samples, n_features = X.shape
        # Randomly initialize cluster centroids
        random_idx = np.random.permutation(n_samples)[:self.k]
        self.centroids = X[random_idx]
        
        for _ in range(self.max_iters):
            # Assign clusters
            closest, _ = pairwise_distances_argmin_min(X, self.centroids)
            new_centroids = np.array([X[closest == i].mean(axis=0) for i in range(self.k)])
            
            # If centroids do not change, then convergence is reached
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
    
    def predict(self, X):
        closest, _ = pairwise_distances_argmin_min(X, self.centroids)
        return closest
    
    def getCost(self):
        closest, _ = pairwise_distances_argmin_min(self.X, self.centroids)
        cost = np.sum([np.sum((self.X[closest == i] - centroid) ** 2) for i, centroid in enumerate(self.centroids)])
        return cost

