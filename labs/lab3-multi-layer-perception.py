import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load datase
file_path = "diabetes.csv"
df = pd.read_csv(file_path)

# Extract the target variable (assuming 'Outcome' is the target)
y = df['Outcome'].values
X = df.drop(columns=['Outcome']).values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Initialize weights and biases for two hidden layers
        self.weights_input_hidden1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size1))
        self.weights_hidden1_hidden2 = np.random.uniform(-1, 1, (self.hidden_size1, self.hidden_size2))
        self.weights_hidden2_output = np.random.uniform(-1, 1, (self.hidden_size2, self.output_size))

        self.bias_hidden1 = np.zeros((1, self.hidden_size1))
        self.bias_hidden2 = np.zeros((1, self.hidden_size2))
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, X):
        self.hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_output1 = sigmoid(self.hidden_input1)

        self.hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_output2 = sigmoid(self.hidden_input2)

        self.final_input = np.dot(self.hidden_output2, self.weights_hidden2_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, learning_rate):
        error = y.reshape(-1, 1) - self.final_output
        d_output = error * sigmoid_derivative(self.final_output)

        error_hidden2 = d_output.dot(self.weights_hidden2_output.T)
        d_hidden2 = error_hidden2 * sigmoid_derivative(self.hidden_output2)

        error_hidden1 = d_hidden2.dot(self.weights_hidden1_hidden2.T)
        d_hidden1 = error_hidden1 * sigmoid_derivative(self.hidden_output1)

        # Update weights and biases
        self.weights_hidden2_output += self.hidden_output2.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        self.weights_hidden1_hidden2 += self.hidden_output1.T.dot(d_hidden2) * learning_rate
        self.bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden1 += X.T.dot(d_hidden1) * learning_rate
        self.bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.abs(y.reshape(-1, 1) - self.final_output))
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def predict(self, X):
        predictions = self.forward(X)
        return predictions.flatten()

# Train the modified MLP with an additional hidden layer
mlp = MLP(input_size=X_train.shape[1], hidden_size1=10, hidden_size2=10, output_size=1)
mlp.train(X_train, y_train, learning_rate=0.1, epochs=10000)

# Make predictions
y_pred = mlp.predict(X_test)

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(y_test[:400], label="Actual", marker='o', linestyle='dashed')
plt.plot(y_pred[:400], label="Predicted", marker='s', linestyle='solid')
plt.xlabel("Time")
plt.ylabel("Outcome")
plt.title("MLP Time-Series Prediction with Two Hidden Layers")
plt.legend()
plt.show()
