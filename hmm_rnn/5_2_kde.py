import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


class KDE:
    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None

    def fit(self, data):
        self.data = np.array(data)

    def _kernel_function(self, u):
        if self.kernel == "gaussian":
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)
        elif self.kernel == "box":
            return 0.5 * (np.abs(u) <= 1)
        elif self.kernel == "triangular":
            return np.maximum(1 - np.abs(u), 0)
        else:
            raise ValueError("Unsupported kernel. Choose 'gaussian', 'box', or 'triangular'.")

    def predict(self, x):
        x = np.array(x)
        n, d = self.data.shape
        density_sum = 0.0

        for xi in self.data:
            u = (x - xi) / self.bandwidth
            kernel_value = np.prod(self._kernel_function(u))  # Product across dimensions
            density_sum += kernel_value

        return density_sum / (n * (self.bandwidth ** d))

    def visualize(self, resolution=100):
        if self.data is None or self.data.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        x_min, y_min = self.data.min(axis=0) - self.bandwidth
        x_max, y_max = self.data.max(axis=0) + self.bandwidth
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        # Parameters
        output_dir = "../../assignments/5/figures"

        density = np.array([
            self.predict([x, y]) for x, y in zip(X.ravel(), Y.ravel())
        ]).reshape(X.shape)

        plt.contourf(X, Y, density, levels=20, cmap="viridis")
        plt.colorbar(label="Density")
        plt.scatter(self.data[:, 0], self.data[:, 1], s=1, c="red", label="Data points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"KDE Density Plot using {self.kernel.capitalize()} Kernel")
        plt.legend()

        # Save the plot in the specified directory
        save_path = os.path.join(output_dir, "kde_density_plot_gaussian")
        plt.savefig(save_path)
        plt.show()





# Parameters for the two clusters
n_large = 3000   # Number of points in the large, diffused circle
n_small = 500    # Number of points in the small, dense circle
radius_large = 2.1 # Approximate radius for the large circle
radius_small = 0.35 # Approximate radius for the small circle
center_large = (0, 0) # Center of the large circle
center_small = (1, 1) # Center of the small circle
noise_std = 0.3  # Standard deviation for noise

# Generate points for the large circle
angles_large = 2 * np.pi * np.random.rand(n_large)
radii_large = radius_large * np.sqrt(np.random.rand(n_large)) # sqrt to spread points uniformly
x_large = center_large[0] + radii_large * np.cos(angles_large)
y_large = center_large[1] + radii_large * np.sin(angles_large)

# Add noise to the large circle
x_large += np.random.normal(0, noise_std, n_large)
y_large += np.random.normal(0, noise_std, n_large)

# Generate points for the small circle
angles_small = 2 * np.pi * np.random.rand(n_small)
radii_small = radius_small * np.sqrt(np.random.rand(n_small)) # sqrt for uniform distribution
x_small = center_small[0] + radii_small * np.cos(angles_small)
y_small = center_small[1] + radii_small * np.sin(angles_small)

# Add noise to the small circle
noise_std1 = 0.05
x_small += np.random.normal(0, noise_std1, n_small)
y_small += np.random.normal(0, noise_std1, n_small)

# Combine the two circles
x = np.concatenate([x_large, x_small])
y = np.concatenate([y_large, y_small])

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=1, color="black")
plt.title("Synthetic Dataset with Overlapping Density Regions and Noise")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(True)
plt.show()


x = np.concatenate([x_large, x_small])
y = np.concatenate([y_large, y_small])
data = np.column_stack((x, y))



# Generate synthetic dataset (code omitted for brevity, see previous code for dataset generation)

# Fit KDE
kde = KDE(bandwidth=0.5, kernel="gaussian")
kde.fit(data)
kde.visualize()




#  Implementing the GMM Class

class GMM:
    def __init__(self, n_components, tol=0.5, max_iter=200):
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





     # Assuming other methods are implemented correctly
    def visualize(self, data):
        plt.scatter(data[:, 0], data[:, 1], s=10, label="Data")

        # Create a grid to evaluate the PDF
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        for k in range(self.n_components):
            plt.scatter(self.means[k, 0], self.means[k, 1], s=100, label=f"GMM Component {k+1}")
            
            # Calculate the PDF for the current component
            rv = multivariate_normal(self.means[k], self.covariances[k])
            Z = rv.pdf(pos)

            # Plot the contour for the PDF
            plt.contour(X, Y, Z, levels=5, cmap="viridis")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        # plt.savefig("component_7")
        plt.show()
        


# Fit GMM with 2 components
gmm = GMM(n_components=2)
gmm.fit(data)
gmm.visualize(data)
