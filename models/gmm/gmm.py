import numpy as np
from scipy.stats import multivariate_normal


#  4.1 Implementing the GMM Class

class GMM:
    def __init__(self, n_components, tol=0.1, max_iter=10):
        self.n_components = n_components  # Number of Gaussian components
        self.tol = tol  # Tolerance to stop the EM algorithm
        self.max_iter = max_iter  # Maximum number of iterations

    def initialize_params(self, X):
        n_samples, n_features = X.shape
        
        # Initialize weights, means, and covariances randomly
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

    def e_step(self, X):
        n_samples = X.shape[0]
        log_likelihoods = np.zeros((n_samples, self.n_components))
        
        # Calculate log membership probabilities (responsibilities)
        for c in range(self.n_components):
            cov = self.covariances[c]
            if np.linalg.det(cov) == 0:  # Ensure covariance is not singular
                cov += np.eye(cov.shape[0]) * 0.1
            log_likelihoods[:, c] = np.log(self.weights[c]) + multivariate_normal.logpdf(X, self.means[c], cov)

        # Normalize to get the responsibilities
        log_likelihoods_max = np.max(log_likelihoods, axis=1, keepdims=True)
        log_likelihoods -= log_likelihoods_max  # For numerical stability
        likelihoods = np.exp(log_likelihoods)
        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
        responsibilities = likelihoods / total_likelihood

        return responsibilities

    def m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Update weights, means, and covariances
        weights_new = np.sum(responsibilities, axis=0) / n_samples
        means_new = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]
        covariances_new = np.zeros((self.n_components, n_features, n_features))
        
        for c in range(self.n_components):
            X_centered = X - means_new[c]
            cov = np.dot(responsibilities[:, c] * X_centered.T, X_centered) / np.sum(responsibilities[:, c])
            # Ensure covariance is not singular
            if np.linalg.det(cov) == 0:
                cov += np.eye(cov.shape[0]) * 0.1
            covariances_new[c] = cov
        
        return weights_new, means_new, covariances_new

    def fit(self, X):
        self.initialize_params(X)
        log_likelihood_old = -np.inf
        iteration = 0

        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self.e_step(X)
            # M-step
            self.weights, self.means, self.covariances = self.m_step(X, responsibilities)
            # Calculate log-likelihood
            log_likelihood_new = self.getLikelihood(X)
            # Print the iteration number and log-likelihood for debugging
            print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood_new}")
            # aic, bic = self.calculate_aic_bic(X)
            # print(f"Iteration {iteration + 1}: aic = {aic}: bic = {bic}")
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                print(f"Convergence reached at iteration {iteration + 1}")
                break
            log_likelihood_old = log_likelihood_new

    def getMembership(self, X):
        return self.e_step(X)

    def getParams(self):
        return {
            'weights': self.weights,
            'means': self.means,
            'covariances': self.covariances
        }

    def getLikelihood(self, X):
        log_likelihood = 0
        for c in range(self.n_components):
            cov = self.covariances[c]
            if np.linalg.det(cov) == 0:  # Ensure covariance is not singular
                cov += np.eye(cov.shape[0]) * 0.1
            log_likelihood += np.sum(np.log(self.weights[c]) + multivariate_normal.logpdf(X, self.means[c], cov))
        return log_likelihood

    def calculate_aic_bic(self, X):
        n_samples, n_features = X.shape
        # Calculate the number of parameters
        n_params = self.n_components * (n_features * (n_features + 3) // 2 + n_features + 1)
        log_likelihood = self.getLikelihood(X)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        
        return aic, bic