# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import wandb  # For experiment tracking




class MLPRegressor:
    def __init__(self, hidden_layers=[9], activation='relu', optimizer='sgd', learning_rate=0.1, 
                 batch_size=32, epochs=100, early_stopping_patience=40, loss_funct='mse'):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weights = []
        self.biases = []
        self.train_losses = []
        self.val_losses = []
        self.loss_funct = loss_funct

    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -709, 709)))  # Clip to avoid overflow
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:  # linear
            return x

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - np.power(x, 2)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        else:  # linear
            return np.ones_like(x)

    def _forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._activation_function(z)
            activations.append(a)
        return activations

    def _backward_propagation(self, X, y, activations):
        m = X.shape[0]
        delta = activations[-1] - y
        dz = delta/m
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True) 
            gradients.append((dW, db))
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self._activation_derivative(activations[i])
        return list(reversed(gradients))

    def _update_parameters(self, gradients):
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def _sgd(self, X, y):
        for i in range(X.shape[0]):
            X_i = X[i:i+1]
            y_i = y[i:i+1]
            activations = self._forward_propagation(X_i)
            gradients = self._backward_propagation(X_i, y_i, activations)
            self._update_parameters(gradients)

    def _batch_gd(self, X, y):
        activations = self._forward_propagation(X)
        gradients = self._backward_propagation(X, y, activations)
        self._update_parameters(gradients)

    def _mini_batch_gd(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]
            activations = self._forward_propagation(X_batch)
            gradients = self._backward_propagation(X_batch, y_batch, activations)
            self._update_parameters(gradients)

    def _loss_function(self, y_true, y_pred):
        if self.loss_funct == 'mse':
            return np.mean((y_true - y_pred)**2)
        elif self.loss_funct == 'bce':
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y, wb=False):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        input_size = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1  # Adjust output size based on y
        self._initialize_parameters(input_size, output_size)
        best_val_loss = float('inf')
        patience_counter = 0
        all_train_losses = []
        all_val_losses = []

        for epoch in range(self.epochs):
            # Training
            if self.optimizer == 'sgd':
                self._sgd(X_train, y_train)
            elif self.optimizer == 'batch':
                self._batch_gd(X_train, y_train)
            elif self.optimizer == 'mini_batch':
                self._mini_batch_gd(X_train, y_train)

            # Calculate losses
            train_activations = self._forward_propagation(X_train)
            train_loss = self._loss_function(y_train, train_activations[-1])

        # Backward propagation (calculate gradients)
            gradients1 = self._backward_propagation(X_train, y_train, train_activations)
            
            # Update parameters using the gradients
            self._update_parameters(gradients1)
               

            val_activations = self._forward_propagation(X_val)
            val_loss = self._loss_function(y_val, val_activations[-1])
            
            gradients2 = self._backward_propagation(X_val, y_val, val_activations)
            
            # Update parameters using the gradients
            self._update_parameters(gradients2)

            
            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)

            if wb: 
                # Log the metrics to the wandb dashboard
                data_to_log = {
                    'epoch': epoch + 1,
                    'learning_rate': self.learning_rate,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "activation": self.activation,
                    "optimizer": self.optimizer,
                }
                wandb.log(data_to_log)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return all_train_losses, all_val_losses

    def predict(self, X):
        activations = self._forward_propagation(X)
        return activations[-1]  # Return continuous values

    def loss_dict(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def _compute_cost(self, y_pred, y_true):
        m = y_true.shape[0]
        cost = -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Cross-entropy cost
        return cost

    def gradient_check(self, X, y, epsilon=1e-7):
        # Compute gradients using backpropagation
        activations = self._forward_propagation(X)
        gradients = self._backward_propagation(X, y, activations)
        
        # Flatten all parameters and gradients
        params = np.concatenate([w.ravel() for w in self.weights] + [b.ravel() for b in self.biases])
        grads = np.concatenate([dw.ravel() for dw, _ in gradients] + [db.ravel() for _, db in gradients])
        
        # Compute numerical gradients
        num_grads = np.zeros_like(params)
        perturb = np.zeros_like(params)
        for i in range(len(params)):
            perturb[i] = epsilon
            params_plus = params + perturb
            self._set_params(params_plus)
            loss_plus = self._compute_cost(self._forward_propagation(X)[-1], y)
            
            params_minus = params - perturb
            self._set_params(params_minus)
            loss_minus = self._compute_cost(self._forward_propagation(X)[-1], y)
            
            num_grads[i] = (loss_plus - loss_minus) / (2 * epsilon)
            perturb[i] = 0
        
        # Reset parameters
        self._set_params(params)
        
        # Compute the relative difference
        diff = np.linalg.norm(num_grads - grads) / (np.linalg.norm(num_grads) + np.linalg.norm(grads))
        
        print(f"Relative difference: {diff}")
        if diff < 1e-7:
            print("Gradient check passed!")
        else:
            print("Gradient check failed.")
        
        return diff

    def _set_params(self, params):
        start = 0
        for i in range(len(self.weights)):
            end = start + self.weights[i].size
            self.weights[i] = params[start:end].reshape(self.weights[i].shape)
            start = end
        for i in range(len(self.biases)):
            end = start + self.biases[i].size
            self.biases[i] = params[start:end].reshape(self.biases[i].shape)
            start = end