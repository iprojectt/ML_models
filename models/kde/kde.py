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


