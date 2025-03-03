import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset with corrected delimiter
url = "https://homepages.ecs.vuw.ac.nz/~marslast/Code/Data/PNoz.dat"
df = pd.read_csv(url, sep=r'\s+', header=None)

# Print dataset structure
print("Dataset Shape:", df.shape)
print("Dataset Sample:")
print(df.head())

# Drop any empty columns
df = df.dropna(axis=1, how='all')

# Convert all columns to numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any remaining NaN rows
df = df.dropna()

# Extract the target variable (assuming the last column is the target)
y = df.iloc[:, -1].values
X = df.iloc[:, :-1].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Normalize the target (y) using MinMaxScaler
y_scaler = MinMaxScaler(feature_range=(-1, 1))
y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Convert to time-series data
k = 3  # Number of past values used as input
tau = 2  # Step size for selecting past values

def create_time_series_data(X, y, k, tau):
    X_new, y_new = [], []
    for i in range(k * tau, len(X)):
        X_new.append(X[i - k * tau:i:tau].flatten())
        y_new.append(y[i])
    return np.array(X_new), np.array(y_new)

X_ts, y_ts = create_time_series_data(X, y, k, tau)

# Split dataset into training and testing sets
split_idx = int(0.8 * len(X_ts))
X_train, X_test = X_ts[:split_idx], X_ts[split_idx:]
y_train, y_test = y_ts[:split_idx], y_ts[split_idx:]

def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def clip_gradients(grad, clip_value=5.0):
    norm = np.linalg.norm(grad)
    if norm > clip_value:
        return grad * (clip_value / norm)
    return grad

class RecurrentMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, weight_decay=0.00005, gradient_clip=5.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # L2 regularization
        self.gradient_clip = gradient_clip  # Gradient clipping
        
        # He Initialization
        limit = np.sqrt(2 / (self.input_size + self.hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-limit, limit, (self.hidden_size, self.output_size))
        self.weights_hidden_hidden = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))  # Recurrent connection
        
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    
    def forward(self, X):
        hidden_state = np.zeros((self.hidden_size,))  # Initialize hidden state as a 1D vector
        final_outputs = []
        
        for i in range(X.shape[0]):
            hidden_state = tanh_activation(
                np.dot(X[i], self.weights_input_hidden) +
                np.dot(hidden_state, self.weights_hidden_hidden) +
                self.bias_hidden.reshape(-1)
            )
            final_output = np.dot(hidden_state, self.weights_hidden_output) + self.bias_output.reshape(-1)
            final_outputs.append(final_output)
        
        return np.array(final_outputs).reshape(-1, 1)
    
    def backward(self, X, y):
        y = y.reshape(-1, 1)
        final_outputs = self.forward(X)
        error = y - final_outputs
        d_output = error * tanh_derivative(final_outputs)
        
        # Apply gradient clipping
        d_output = clip_gradients(d_output)
        
        # Update hidden layer weights
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * tanh_derivative(X @ self.weights_input_hidden)
        d_hidden = clip_gradients(d_hidden)
        
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, d_hidden) / d_hidden.shape[0]
        self.weights_hidden_hidden += self.learning_rate * np.dot(d_hidden.T, d_hidden) / d_hidden.shape[0]
        self.weights_hidden_output += self.learning_rate * np.dot(d_hidden.T, d_output) / d_output.shape[0]
        self.bias_hidden += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / d_hidden.shape[0]
        self.bias_output += self.learning_rate * np.sum(d_output, axis=0, keepdims=True) / d_output.shape[0]
    
    def train(self, X, y, epochs=10000, early_stopping_patience=10000):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.backward(X, y)
            loss = np.mean(np.abs(y.reshape(-1, 1) - self.forward(X)))
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    def predict(self, X):
        predictions = self.forward(X)
        return y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Train Recurrent MLP on the dataset
rmlp = RecurrentMLP(input_size=X_train.shape[1], hidden_size=100, output_size=1, learning_rate=0.001, weight_decay=0.00005, gradient_clip=5.0)
rmlp.train(X_train, y_train, epochs=10000)

# Make predictions
y_pred = rmlp.predict(X_test)

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(y_test[:400], label="Actual", marker='o', linestyle='dashed')
plt.plot(y_pred[:400], label="Predicted", marker='s', linestyle='solid')
plt.xlabel("Time")
plt.ylabel("Outcome")
plt.title("Recurrent MLP Time-Series Prediction - Palmerston North Ozone Data")
plt.legend()
plt.show()
