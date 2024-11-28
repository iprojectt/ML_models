import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.all_eigenvalues_ = None

    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Store all eigenvalues for scree plot
        self.all_eigenvalues_ = eigenvalues
        # Select the top n_components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components_.T) + self.mean_

    def checkPCA(self, X, tolerance=6):
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        # Calculate reconstruction error
        error = np.linalg.norm(X - X_reconstructed)
        
        # Return True if the reconstruction error is within the tolerance
        return error < tolerance
    
    def plot_scree_plot(self, eigenvalues):
    # Number of principal components
        n_components = len(eigenvalues)
        
        # Plot the scree plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n_components + 1), eigenvalues, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        plt.show()
