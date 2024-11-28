# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Load the dataset
df = pd.read_csv('WineQT.csv')

# 1. Check for missing values or inconsistent data
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# If there are any missing values, decide how to handle them (e.g., fill with mean/median or drop rows)
# For now, we'll fill missing numerical values with the mean of the respective column
df.fillna(df.mean(), inplace=True)

# 2. Describe the dataset after handling missing values
description = df.describe()
# print("Dataset description:\n", description)
print("description",description)

# 3. Plot distribution for all features
# We'll exclude the 'Id' column for this plot
columns_to_plot = df.columns.drop('Id')

plt.figure(figsize=(15, 10))  # Adjust the figure size for better readability
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 3, i)  # Create subplots in a 4x3 grid
    plt.hist(df[column], bins=20, edgecolor='black')
    plt.title(f'{column} Distribution')
    plt.grid(True)

plt.tight_layout()
plt.show()


# 4. Normalize and standardize the data
# Dropping the 'Id' column as it's not needed for normalization/standardization
df_features = df.drop(['Id'], axis=1)

# Normalize (MinMaxScaler)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df_features), columns=df_features.columns)

# Standardize (StandardScaler)
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df_features), columns=df_features.columns)

# Display normalized and standardized data

print("\nStandardized Data:")
df_standardized.head()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess the WineQT dataset
df = pd.read_csv('WineQT.csv')
X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert y to one-hot encoded format
y_encoded = pd.get_dummies(y).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

class MLPClassifier:
    def __init__(self, hidden_layers=[9], activation='relu', optimizer='sgd', learning_rate=0.1, 
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

    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
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
        # Apply softmax on the output layer for multi-class classification
        activations[-1] = self._softmax(activations[-1])
        return activations

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _backward_propagation(self, X, y, activations):
        m = X.shape[0]
        delta = activations[-1] - y  # Using cross-entropy, this is the gradient for the softmax output layer
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

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        input_size = X.shape[1]
        output_size = y.shape[1]
        self._initialize_parameters(input_size, output_size)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            # Training
            if self.optimizer == 'sgd':
                self._sgd(X_train, y_train)
            elif self.optimizer == 'batch':
                self._batch_gd(X_train, y_train)
            elif self.optimizer == 'mini_batch':
                self._mini_batch_gd(X_train, y_train)

            # Calculate losses using cross-entropy
            train_activations = self._forward_propagation(X_train)
            train_loss = self._compute_cost(train_activations[-1], y_train)
            val_activations = self._forward_propagation(X_val)
            val_loss = self._compute_cost(val_activations[-1], y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        return self

    def predict(self, X):
        activations = self._forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

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

# Train the model
mlp = MLPClassifier(hidden_layers=[9], activation='relu', optimizer='mini_batch', learning_rate=0.01, epochs=100, batch_size=32)
mlp.fit(X_train, y_train)

# Make predictions
predictions = mlp.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, predictions)
print(f"Accuracy: {accuracy}")

# Gradient checking
print("Performing gradient check...")
mlp.gradient_check(X_train[:20], y_train[:20])



# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import wandb  # For experiment tracking
from sklearn.metrics import log_loss



# Load the dataset
df = pd.read_csv('WineQT.csv')
# Preprocess the data
X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Convert y to one-hot encoded format
y_encoded = pd.get_dummies(y).values
print(df.isnull().sum())  # Check for NaN values in the DataFrame


class MLPClassifier:
    def __init__(self, hidden_layers=[9], activation='relu', optimizer='sgd', learning_rate=0.1, 
                 batch_size=32, epochs=100, early_stopping_patience=40):
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
        

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]

            a = self._activation_function(z)
            activations.append(a)
        # Apply softmax activation to the output layer
        activations[-1] = self._softmax(activations[-1])
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

    def fit(self, X, y,wb=False):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        input_size = X.shape[1]
        output_size = y.shape[1]
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

            # Calculate losses using cross-entropy
            train_activations = self._forward_propagation(X_train)
            train_loss = log_loss(y_train, train_activations[-1])

        # Backward propagation (calculate gradients)
            gradients1 = self._backward_propagation(X_train, y_train, train_activations)
            
            # Update parameters using the gradients
            self._update_parameters(gradients1)
                
            val_activations = self._forward_propagation(X_val)
            val_loss = log_loss(y_val, val_activations[-1])



            gradients2 = self._backward_propagation(X_val, y_val, val_activations)
            
            # Update parameters using the gradients
            self._update_parameters(gradients2)


            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)


            predictions = self.predict(X_val)
            y_test_labels = np.argmax(y_val, axis=1)
            accuracy = accuracy_score(y_test_labels, predictions)
            f1 = f1_score(y_test_labels, predictions, average='weighted',zero_division=1)
            precision = precision_score(y_test_labels, predictions, average='weighted',zero_division=1)
            recall = recall_score(y_test_labels, predictions, average='weighted',zero_division=1)

            if wb: 
            # create a dictionary where every entry in it is a metric at a given point in time 
                data_to_log = {
                    'epoch': epoch + 1,
                    'learning_rate': self.learning_rate,

                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "activation": self.activation,
                    "optimizer": self.optimizer,

                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                }
                # Log the metrics to the wandb dashboard
                wandb.log(data_to_log)
                    # Log metrics to wandb

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
        return np.argmax(activations[-1], axis=1)



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




# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
# Define sweep configurations
sweep_configurations = {
    "method": "grid",
    "name": "Hyperparameter Tuning",
    "metric": {
        "goal": "maximize",
        "name": "accuracy"
    },
    "parameters": {
        "epochs": {
            "values": [500,1000]
        },
        "batch_size": {
            "values": [32]
        },
        "learning_rates": {
            "values": [0.01]
        },
        "activations": {
            "values": ["relu", "sigmoid", "tanh","linear"]
        },
        "hidden_layers_options": { 
            "values": [[9],[64,32]]
        },        
        "optimizers": {
            "values": ["batch"]
        }
    }
}





def sweep_agent_manager():
    # Initialize a new sweep
    wandb.init(project="mlp-classifier")
    # get the configuration of the run
    config = dict(wandb.config)
    epochs = config["epochs"]
    hidden_layers_options = config["hidden_layers_options"]
    activations = config["activations"]
    optimizers = config["optimizers"]
    learning_rates = config["learning_rates"]
    batch_size = config["batch_size"]
    # set the name of the run
    run_name = f"Demo_{activations}_optimizers{optimizers}_hidden_layers_options{hidden_layers_options}_epox{epochs}_lr{learning_rates}_batch_size{batch_size}"
    wandb.run.name = run_name
    # Train and evaluate the model
    mlp = MLPClassifier(hidden_layers=hidden_layers_options, activation=activations, optimizer=optimizers, learning_rate=learning_rates, epochs=epochs, batch_size=batch_size)

    # Fit both models
    train_losses_mse, val_losses_mse = mlp.fit(X_train, y_train, wb=True)

    # Remove the plotting code from here

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_configurations, project="mlp-classifier")
# Run the agent
wandb.agent(sweep_id, function=sweep_agent_manager,count=2)
wandb.finish()

import wandb
# Load all the runs from the project
api = wandb.Api()
runs = api.runs("mlp-classifier")
# Initialize variables to store the best results
best_accuracy = 0
best_run = None
# Iterate through all the runs to find the one with the highest accuracy
for run in runs:
    # Extract the accuracy metric
    accuracy = run.summary.get("accuracy", 0)
    # Check if this run has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_run = run
# Display the best performing hyperparameters
if best_run:
    print(f"Best run ID: {best_run.id}")
    print(f"Best accuracy: {best_accuracy}")
    print("Best hyperparameters:")
    for key, value in best_run.config.items():
        print(f"{key}: {value}")
else:
    print("No runs found with accuracy metric.")




# learning_rates = [0.001, 0.01, 0.1, 0.5]
# losses_per_lr = {}

# for learning_rates in learning_rates:
#     # Initialize and train model with varying learning rates
#     mlp = MLPClassifier(hidden_layers=[34,52], activation='tanh', optimizer='sgd', learning_rate=learning_rates, epochs=100, batch_size=32)
#     train_losses1 = mlp.fit(X_train, y_train)
    
#     # Store the validation loss over epochs
#     losses_per_lr[learning_rates] = train_losses1

# # Plot loss vs. epochs for each learning rate
# plt.figure(figsize=(10, 6))
# for learning_rates, losses in losses_per_lr.items():
#     plt.plot(losses, label=f'LR: {learning_rates}')
# plt.title("Effect of Learning Rate on Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Validation Loss")
# plt.legend()
# plt.show()



batch_sizes = [16, 32, 64, 128]
losses_per_batch = {}

for batch_size in batch_sizes:
    # Initialize and train model with varying batch sizes
    mlp = MLPClassifier(hidden_layers=[34,52], activation='tanh', optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=batch_size)
    train_losses3 = mlp.fit(X_train, y_train)
    
    # Store the validation loss over epochs
    losses_per_batch[batch_size] = train_losses3

# Plot loss vs. epochs for each batch size
plt.figure(figsize=(10, 6))
for batch_size, losses2 in losses_per_batch.items():
    plt.plot(losses2, label=f'Batch Size: {batch_size}')
plt.title("Effect of Batch Size on Loss")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()


# activations = ['relu', 'sigmoid', 'tanh', 'linear']
# losses_per_activation = {}

# for activation in activations:
#     # Initialize and train model with varying activation functions
#     mlp = MLPClassifier(hidden_layers=[34,52], activation=activations, optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=32)
#     train_losses2 = mlp.fit(X_train, y_train)
    
#     # Store the validation loss over epochs
#     losses_per_activation[activations] = train_losses2

# # Plot loss v
# # Plot loss vs. epochs for each activation function
# plt.figure(figsize=(10, 6))
# for activation, losses in losses_per_activation.items():
#     plt.plot(losses, label=activation)
# plt.title("Effect of Non-linearity (Activation Functions) on Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Validation Loss")
# plt.legend()
# plt.show()





import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
import wandb

# Load the dataset
df = pd.read_csv('advertisement.csv')

# Preprocess the data
# Convert categorical features to numerical (if necessary)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['education'] = df['education'].astype('category').cat.codes
df['city'] = df['city'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
df['most bought item'] = df['most bought item'].astype('category').cat.codes

# Features and labels
X = df.drop(['labels'], axis=1)
y = df['labels'].str.split()  # Split labels into lists

# Use MultiLabelBinarizer to one-hot encode the labels
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

# Calculate the number of unique labels
num_labels = len(mlb.classes_)  # Get the number of unique labels

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


class MultiLabelMultiLabelMLPClassifier:
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
            y_test_labels = y_val
            accuracy = self.accuracy_score_TP(y_test_labels, predictions)
            f1 = f1_score(y_test_labels, predictions, average='weighted',zero_division=1)
            precision = precision_score(y_test_labels, predictions, average='weighted',zero_division=1)
            recall = recall_score(y_test_labels, predictions, average='weighted',zero_division=1)



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

    def predict(self, X, threshold=0.46):
        activations = self._forward_propagation(X)
        return (activations[-1] >= threshold).astype(int)
    

    def accuracy_score_TP(self,y_test,y_pred):
        

            # Calculate true positives, true negatives, false positives, false negatives
        TP = np.sum((y_pred == 1) & (y_test == 1))
        TN = np.sum((y_pred == 0) & (y_test == 0))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        FN = np.sum((y_pred == 0) & (y_test == 1))

        # Calculate accuracy based on these counts
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        return accuracy

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)




sweep_configurations = {
    "method": "grid",
    "name": "Hyperparameter Tuning",
    "metric": {
        "goal": "minimize",
        "name": "val_loss"
    },
    "parameters": {
        "epochs": {
            "values": [500,1000]
        },
        "batch_size": {
            "values": [32,60]
        },
        "learning_rates": {
            "values": [0.001, 0.01, 0.1]
        },
        "activations": {
            "values": ["relu", "sigmoid", "tanh"]
        },
        "hidden_layers_options": { 
            "values": [[9],[34,52],[9,32,64]]
        },        
        "optimizers": {
            "values": ["sgd","mini_batch","adam"]
        }
    }
}





def sweep_agent_manager():
    # Initialize a new sweep
    wandb.init(project="multi-label-classifier")
   # get the configuration of the run
    config = dict(wandb.config)
    epochs = config["epochs"]
    hidden_layers_options = config["hidden_layers_options"]
    activations = config["activations"]
    optimizers = config["optimizers"]
    learning_rates = config["learning_rates"]
    batch_size = config["batch_size"]
    # set the name of the run
    run_name = f"Demo_{activations}_optimizers{optimizers}_hidden_layers_options{hidden_layers_options}_epox{epochs}_lr{learning_rates}_batch_size{batch_size}"
    wandb.run.name = run_name
# Train and evaluate the model
    mlp = MultiLabelMultiLabelMLPClassifier(hidden_layers=[32, 16], activation='relu', optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=32)
    mlp.fit(X_train, y_train,wb=True)

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_configurations, project="multi-label-classifier")
# Run the agent
wandb.agent(sweep_id, function=sweep_agent_manager,count=5)
wandb.finish()





import wandb
# Load all the runs from the project
api = wandb.Api()
runs = api.runs("multi-label-classifier")
# Initialize variables to store the best results
best_accuracy = 0
best_run = None
# Iterate through all the runs to find the one with the highest accuracy
for run in runs:
    # Extract the accuracy metric
    accuracy = run.summary.get("accuracy", 0)
    # Check if this run has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_run = run
# Display the best performing hyperparameters
if best_run:
    print(f"Best run ID: {best_run.id}")
    print(f"Best accuracy: {best_accuracy}")
    print("Best hyperparameters:")
    for key, value in best_run.config.items():
        print(f"{key}: {value}")
else:
    print("No runs found with accuracy metric.")




# # Define and initialize wandb
# wandb.init(project="multi-label-classifier")

# # Create and train the model
# mlp = MultiLabelMultiLabelMLPClassifier(hidden_layers=[32, 16], activation='relu', optimizer='sgd', learning_rate=0.01, epochs=100, batch_size=32)
# train_losses = mlp.fit(X_train, y_train, wb=True)




# # Evaluate the model for different thresholds
# # Evaluate the model for different thresholds


# y_pred = mlp.predict(X_test, threshold=0.46)

# # Calculate true positives, true negatives, false positives, false negatives
# TP = np.sum((y_pred == 1) & (y_test == 1))
# TN = np.sum((y_pred == 0) & (y_test == 0))
# FP = np.sum((y_pred == 1) & (y_test == 0))
# FN = np.sum((y_pred == 0) & (y_test == 1))

# # Calculate accuracy based on these counts
# accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

# # Print the output and actual labels
# print("Predicted outputs using the best threshold:")
# print(y_pred)
# print("\nActual labels:")
# print(y_test)

# # Print the best threshold and accuracy
# print(f"Best accuracy: {accuracy}")

# # Print additional metrics
# print(f"True Positives: {TP}, True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}")


# # # Evaluate the model
# # y_pred = mlp.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)
# # f1 = f1_score(y_test, y_pred, average='weighted')
# # precision = precision_score(y_test, y_pred, average='weighted')
# # recall = recall_score(y_test, y_pred, average='weighted')
# # # Print the output and actual labels
# # print("Predicted outputs:")
# # print(y_pred)
# # print("\nActual labels:")
# # print(y_test)
# # print(f"Accuracy: {accuracy}")
# # print(f"F1 Score: {f1}")
# # print(f"Precision: {precision}")
# # print(f"Recall: {recall}")

# wandb.finish()




# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import wandb  # For experiment tracking


# Load the dataset
data = pd.read_csv('diabetes.csv')

# Preprocess the data
# Handle missing values by filling with the mean or dropping rows
data = data.replace('NA', np.nan).astype(float)
data.fillna(data.mean(), inplace=True)  # Filling missing values with mean

# Define features (X) and target (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Convert y to one-hot encoded format
print(data.isnull().sum()) 

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.reshape(-1, 1), test_size=0.2, random_state=42)

# Define sweep configurations for hyperparameter tuning
sweep_configurations = {
    "method": "grid",
    "name": "Hyperparameter Tuning for MLP Regression",
    "metric": {
        "goal": "maximize",
        "name": "r2_score"  # Using R2 score as the evaluation metric
    },
    "parameters": {
        "epochs": {
            "values": [500, 1000]
        },
        "batch_size": {
            "values": [32, 60]
        },
        "learning_rates": {
            "values": [0.001, 0.01, 0.1]
        },
        "activations": {
            "values": ["relu", "sigmoid", "tanh"]
        },
        "hidden_layers_options": { 
            "values": [[9], [34, 52], [9, 32, 64]]
        },        
        "optimizers": {
            "values": ["sgd", "mini_batch", "adam"]
        }
    }
}

def sweep_agent_manager():
    # Initialize a new sweep
    wandb.init(project="mlp-regressor-final")
    # Get the configuration of the run
    config = dict(wandb.config)
    epochs = config["epochs"]
    hidden_layers_options = config["hidden_layers_options"]
    activations = config["activations"]
    optimizers = config["optimizers"]
    learning_rates = config["learning_rates"]
    batch_size = config["batch_size"]
    # Set the name of the run
    run_name = f"Demo_{activations}_optimizers{optimizers}_hidden_layers_options{hidden_layers_options}_epochs{epochs}_lr{learning_rates}"
    wandb.run.name = run_name

    model = MLPRegressor(hidden_layers=hidden_layers_options, 
                         activation=activations,
                         optimizer=optimizers, 
                         learning_rate=learning_rates, 
                         batch_size=batch_size, 
                         epochs=epochs, 
                         early_stopping_patience=40)
    
    # Fit the model and log the results
    train_losses, val_losses = model.fit(X_train, y_train, wb=True)

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_configurations, project="mlp-regressor-final")
# Run the agent
wandb.agent(sweep_id, function=sweep_agent_manager,count=2)
wandb.finish()

# Train two models with different loss functions
mlp_mse = MLPRegressor(hidden_layers=[9], activation='relu', optimizer='mini_batch', learning_rate=0.001, epochs=1000, batch_size=32, loss_funct='mse')
mlp_bce = MLPRegressor(hidden_layers=[9], activation='relu', optimizer='mini_batch', learning_rate=0.001, epochs=1000, batch_size=32, loss_funct='bce')

# Fit both models
train_losses_mse, val_losses_mse = mlp_mse.fit(X_train, y_train)
train_losses_bce, val_losses_bce = mlp_bce.fit(X_train, y_train)

# Plot loss vs epochs for both models
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses_mse, label='Train Loss (MSE)')
plt.plot(val_losses_mse, label='Validation Loss (MSE)')
plt.title('MSE Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses_bce, label='Train Loss (BCE)')
plt.plot(val_losses_bce, label='Validation Loss (BCE)')
plt.title('BCE Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Observations and differences
print("Observations and differences:")
print("1. MSE Loss:")
print(f"   - Final training loss: {train_losses_mse[-1]:.4f}")
print(f"   - Final validation loss: {val_losses_mse[-1]:.4f}")
print("2. BCE Loss:")
print(f"   - Final training loss: {train_losses_bce[-1]:.4f}")
print(f"   - Final validation loss: {val_losses_bce[-1]:.4f}")
print("\nConvergence observations:")
print("- MSE loss tends to converge faster and more smoothly.")
print("- BCE loss may show more fluctuations and slower convergence.")
print("- The scale of losses differs between MSE and BCE due to their different formulations.")
print("- BCE loss might be more sensitive to outliers compared to MSE loss.")
print("- The choice between MSE and BCE depends on the specific problem and desired properties of the model.")

# Gradient checking
print("\nPerforming gradient check for MSE model...")
mlp_mse.gradient_check(X_train[:20], y_train[:20])

print("\nPerforming gradient check for BCE model...")
mlp_bce.gradient_check(X_train[:20], y_train[:20])











# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import wandb  # For experiment tracking
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split




import numpy as np
import pandas as pd

class Knn:

    def __init__(self, k=3, distance_metric='euclidean'):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.distance_metric == 'euclidean':
            distances = self.euclidean_distance(X_test)
        elif self.distance_metric == 'manhattan':
            distances = self.manhattan_distance(X_test)
        elif self.distance_metric == 'cosine':
            distances = self.cosine_distance(X_test)

        # Get the indices of the k nearest neighbors
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Gather the nearest neighbors' labels
        neighbors_labels = self.y_train[neighbors_indices]

        # Get the most common label (majority vote) for each test instance
        y_pred = np.array([self.majority_vote(labels) for labels in neighbors_labels])

        return y_pred

    def manhattan_distance(self, X_test, batch_size=500):
        # X_test = X_test.astype(np.float64)
        num_test_samples = X_test.shape[0]
        num_train_samples = self.X_train.shape[0]
        dists = np.zeros((num_test_samples, num_train_samples))
        
        # Process in batches
        for start_idx in range(0, num_test_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_test_samples)
            X_test_batch = X_test[start_idx:end_idx]
            
            # Compute distances for the current batch
            dists_batch = np.sum(np.abs(X_test_batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
            dists[start_idx:end_idx] = dists_batch
        
        return dists




    def majority_vote(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count_index = np.argmax(counts)
        return unique_labels[max_count_index]










def generate_random_dataset(n_samples=1000, n_features=20, random_state=42):
    np.random.seed(random_state)
    
    # Generate some underlying latent factors
    latent_factors = np.random.randn(n_samples, 5)
    
    # Generate features based on these latent factors
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        weights = np.random.randn(5)
        noise = np.random.randn(n_samples) * 0.1
        X[:, i] = np.dot(latent_factors, weights) + noise
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Generate random dataset
X_random = generate_random_dataset(n_samples=1000, n_features=20)




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


# Define sweep configurations for hyperparameter tuning
sweep_configurations = {
    "method": "grid",
    "name": "Hyperparameter Tuning for MLP Regression",
    "metric": {
        "goal": "maximize",
        "name": "r2_score"  # Using R2 score as the evaluation metric
    },
    "parameters": {
        "epochs": {
            "values": [500, 1000]
        },
        "batch_size": {
            "values": [32, 60]
        },
        "learning_rates": {
            "values": [0.001, 0.01, 0.1]
        },
        "activations": {
            "values": ["relu", "sigmoid", "tanh"]
        },
        "hidden_layers_options": { 
            "values": [[9], [34, 52], [9, 32, 64]]
        },        
        "optimizers": {
            "values": ["sgd", "mini_batch", "adam"]
        }
    }
}





















import numpy as np

class AutoEncoder:
    def __init__(self, input_dim, hidden_layers, latent_dim, activation='relu', 
                 learning_rate=0.01, epochs=1000, batch_size=50, optimizer='sgd'):
        # Model parameters
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_type = activation
        self.optimizer = optimizer

        # Encoder and Decoder architecture
        self.encoder_layers = [input_dim] + hidden_layers + [latent_dim]
        self.decoder_layers = [latent_dim] + hidden_layers[::-1] + [input_dim]

        # Initialize encoder weights and biases
        self.encoder_weights = [
            np.random.randn(self.encoder_layers[i], self.encoder_layers[i + 1]) * 0.01
            for i in range(len(self.encoder_layers) - 1)
        ]
        self.encoder_biases = [np.zeros((1, self.encoder_layers[i + 1])) for i in range(len(self.encoder_layers) - 1)]

        # Initialize decoder weights and biases
        self.decoder_weights = [
            np.random.randn(self.decoder_layers[i], self.decoder_layers[i + 1]) * 0.01
            for i in range(len(self.decoder_layers) - 1)
        ]
        self.decoder_biases = [np.zeros((1, self.decoder_layers[i + 1])) for i in range(len(self.decoder_layers) - 1)]

        # Set activation functions
        self.activation = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)

    def _get_activation_function(self, activation):
        """Returns the appropriate activation function."""
        if activation == 'relu':
            return self.relu
        elif activation == 'sigmoid':
            return self.sigmoid
        elif activation == 'tanh':
            return self.tanh
        return self.linear  # Default to linear activation

    def _get_activation_derivative(self, activation):
        """Returns the appropriate activation derivative."""
        if activation == 'relu':
            return self.relu_derivative
        elif activation == 'sigmoid':
            return self.sigmoid_derivative
        elif activation == 'tanh':
            return self.tanh_derivative
        return self.linear_derivative

    def forward(self, X):
        """Forward pass through encoder and decoder."""
        self.encoder_activations = [X]
        self.decoder_activations = []

        # Encoder forward pass
        for i in range(len(self.encoder_weights)):
            z = np.dot(self.encoder_activations[-1], self.encoder_weights[i]) + self.encoder_biases[i]
            a = self.activation(z)
            self.encoder_activations.append(a)

        # Decoder forward pass
        for i in range(len(self.decoder_weights)):
            z = np.dot(self.decoder_activations[i - 1] if i > 0 else self.encoder_activations[-1], 
                       self.decoder_weights[i]) + self.decoder_biases[i]
            a = z if i == len(self.decoder_weights) - 1 else self.activation(z)
            self.decoder_activations.append(a)

        return self.decoder_activations[-1]

    def backward(self, X, y):
        """Backward pass to compute gradients."""
        gradients_encoder_weights = [np.zeros_like(w) for w in self.encoder_weights]
        gradients_encoder_biases = [np.zeros_like(b) for b in self.encoder_biases]
        gradients_decoder_weights = [np.zeros_like(w) for w in self.decoder_weights]
        gradients_decoder_biases = [np.zeros_like(b) for b in self.decoder_biases]

        # Compute error at output layer
        error = self.decoder_activations[-1] - y

        # Backpropagate through decoder
        for i in reversed(range(len(self.decoder_weights))):
            delta = error if i == len(self.decoder_weights) - 1 else delta.dot(self.decoder_weights[i + 1].T) * self.activation_derivative(self.decoder_activations[i])
            input_to_decoder = self.decoder_activations[i - 1] if i > 0 else self.encoder_activations[-1]
            gradients_decoder_weights[i] = np.dot(input_to_decoder.T, delta) / X.shape[0]
            gradients_decoder_biases[i] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]

        # Backpropagate through encoder
        delta = delta.dot(self.decoder_weights[0].T) * self.activation_derivative(self.encoder_activations[-1])
        for i in reversed(range(len(self.encoder_weights))):
            gradients_encoder_weights[i] = np.dot(self.encoder_activations[i].T, delta) / X.shape[0]
            gradients_encoder_biases[i] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
            if i != 0:
                delta = delta.dot(self.encoder_weights[i].T) * self.activation_derivative(self.encoder_activations[i])

        return gradients_encoder_weights, gradients_encoder_biases, gradients_decoder_weights, gradients_decoder_biases

    def update_parameters(self, gradients_encoder_weights, gradients_encoder_biases, 
                          gradients_decoder_weights, gradients_decoder_biases):
        """Update weights and biases using gradient descent."""
        clip_value = 5.
        for i in range(len(self.encoder_weights)):
            np.clip(gradients_encoder_weights[i], -clip_value, clip_value, out=gradients_encoder_weights[i])          
            self.encoder_weights[i] -= self.learning_rate * gradients_encoder_weights[i]
            np.clip(gradients_encoder_biases[i], -clip_value, clip_value, out=gradients_encoder_biases[i])
            self.encoder_biases[i] -= self.learning_rate * gradients_encoder_biases[i]

        for i in range(len(self.decoder_weights)):
            np.clip(gradients_decoder_weights[i], -clip_value, clip_value, out=gradients_decoder_weights[i])    
            self.decoder_weights[i] -= self.learning_rate * gradients_decoder_weights[i]
            np.clip(gradients_decoder_biases[i], -clip_value, clip_value, out=gradients_decoder_biases[i])
            self.decoder_biases[i] -= self.learning_rate * gradients_decoder_biases[i]

    def train(self, X):
        """Train the autoencoder."""
        for epoch in range(1, self.epochs + 1):
            y_pred = self.forward(X)  # Forward pass
            gradients = self.backward(X, X)  # Backward pass (target is input itself)
            self.update_parameters(*gradients)  # Update weights and biases

            # Compute loss and print every 100 epochs
            loss = self.compute_loss(X, y_pred)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.6f}")

    def compute_loss(self, X, y_pred):
        """Compute MSE loss."""
        return np.mean(np.square(X - y_pred))

    def encode(self, X):
        """Encode input data into the latent space."""
        activations = X
        for i in range(len(self.encoder_weights)):
            z = np.dot(activations, self.encoder_weights[i]) + self.encoder_biases[i]
            activations = self.activation(z)
        return activations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, activated_x):
        return activated_x * (1 - activated_x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, activated_x):
        return np.where(activated_x > 0, 1, 0)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, activated_x):
        return 1 - np.power(activated_x, 2)

    def linear(self, x):
        return x

    def linear_derivative(self, activated_x):
        return np.ones_like(activated_x)











df = pd.read_csv('dataset.csv')
df = df.iloc[:,1:]


df = df.drop_duplicates(subset='track_id')
print(df.shape)
df = df.iloc[:,4:]
df['track_genre'] = pd.factorize(df['track_genre'])[0]
df['explicit'] = pd.factorize(df['explicit'])[0]
df = df.sample(n=50000, random_state=0)
Z = df.iloc[:, 0:16].values

def safe_standardize(Z):
    mean = np.mean(Z, axis=0)
    std_dev = np.std(Z, axis=0)
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    return (Z - mean) / (std_dev + epsilon)

Z_standardized = safe_standardize(Z)
X = Z_standardized[:, :15]  # X will be the standardized values from column 0 to 14
y = Z_standardized[:, -1]   
column_names = df.columns[:15] 
print("NaN values in X:", np.isnan(X).any())
print("Infinite values in X:", np.isinf(X).any())
print(X.shape)
print(y.shape)
# Convert the standardized array (X and y) back to a DataFrame



input_size = X.shape[1]  # Input size
hidden_layers_output = [10,11]
latent_dim = 10


autoencoder = AutoEncoder(
    input_dim=input_size,
    hidden_layers=hidden_layers_output,
    latent_dim=latent_dim,
    activation='relu',
    learning_rate=0.001,
    epochs=1000,
    batch_size=32
)

# Train the Autoencoder
autoencoder.train(X)

# Get the latent representation (encoded features)
latent_representation = autoencoder.encode(X)

# Output the latent representation
print(latent_representation)
print(latent_representation.shape)

# # X = spotify_data.drop('track_genre', axis=1).values
# # y = spotify_data['track_genre'].values

# ## use MLP to classify the data

X = latent_representation
y = y



# # take first 10000 samples

# X = X[:2]
# y = y[:2]

def train_test_val_split(X, y, test_size=0.1, val_size=0.1, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    validate_indices = indices[test_set_size : 2 * test_set_size]
    train_indices = indices[2* test_set_size:]
    return X[train_indices], X[validate_indices], X[test_indices], y[train_indices], y[validate_indices], y[test_indices]

X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X, y, test_size=0.1, val_size=0.1)

print(X_train.shape, X_val.shape, X_test.shape)
print(Z_standardized.shape)
print(X.shape)






# # Initialize and train the MLP model


import time
accuracy_results = []
distance_metrics = ['manhattan']
for distance_metric in distance_metrics:
    print(f"Distance Metric: {distance_metric}")
    start_time = time.time()
    apnaKnn = Knn(k=29, distance_metric=distance_metric)
    apnaKnn.fit(X_train, y_train)
    y_pred1 = apnaKnn.predict(X_test)
    accuracy = np.mean(y_pred1 == y_test)   # Calculate accuracy
    accuracy_results.append((accuracy, 29, distance_metric))
    print(f".k: {29}, accuracy: {accuracy} ")
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds") # Print execution time
# Measure the time for the complete dataset
start_time_complete = time.time()
# mlp.gradient_check(X_train, y_train, epsilon=1e-7, tolerance=1e-3, correct_weights=True)






