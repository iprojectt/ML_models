import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, hamming_loss
import wandb



class MultiLabelMLPClassifier:
    def __init__(self, hidden_layers=[9], activation='relu', optimizer='sgd', learning_rate=0.01,
                 batch_size=32, epochs=100, early_stopping_patience=10):
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

    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -709, 709)))
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
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(activations[i])
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

    def fit(self, X, y, wb=False):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        input_size = X.shape[1]
        output_size = num_labels  # Use calculated number of unique labels
        self._initialize_parameters(input_size, output_size)
        best_val_loss = float('inf')
        patience_counter = 0
        all_train_losses = []

        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                self._sgd(X_train, y_train)
            elif self.optimizer == 'batch':
                self._batch_gd(X_train, y_train)
            elif self.optimizer == 'mini_batch':
                self._mini_batch_gd(X_train, y_train)

            # Calculate losses using binary cross-entropy
            train_activations = self._forward_propagation(X_train)
            train_loss = log_loss(y_train, train_activations[-1])
            val_activations = self._forward_propagation(X_val)
            val_loss = log_loss(y_val, val_activations[-1])
            all_train_losses.append(train_loss)
            
            predictions = self.predict(X_val)
            y_test_labels = np.argmax(y_val, axis=1)
            accuracy = accuracy_score(y_test_labels, predictions)
            f1 = f1_score(y_test_labels, predictions, average='weighted',zero_division=1)
            precision = precision_score(y_test_labels, predictions, average='weighted',zero_division=1)
            recall = recall_score(y_test_labels, predictions, average='weighted',zero_division=1)
            hamming = hamming_loss(y_test_labels, predictions)  

            if wb:
                data_to_log = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,

                    'learning_rate': self.learning_rate,

                    "activation": self.activation,
                    "optimizer": self.optimizer,

                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
            

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

        return all_train_losses

    def predict(self, X, threshold=0.3):
        activations = self._forward_propagation(X)
        return (activations[-1] >= threshold).astype(int)
