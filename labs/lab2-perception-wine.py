import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Load Wine Quality Dataset ===
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(url, delimiter=';')

# Convert wine quality into binary classification:
# Good wine (quality >= 6) -> 1, Bad wine (quality < 6) -> 0
wine_data["quality"] = wine_data["quality"].apply(lambda x: 1 if x >= 6 else 0)

# Extract feature inputs (first 11 columns) and target outputs (last column)
inputs = wine_data.iloc[:, :-1].values  # Features
targets = wine_data.iloc[:, -1].values.reshape(-1, 1)  # Targets

# Normalize features to range [0,1]
inputs = (inputs - inputs.min(axis=0)) / (inputs.max(axis=0) - inputs.min(axis=0))

# === Step 2: Define Perceptron Class (Sequential Update) ===
class PerceptronSequential:
    """ A basic Perceptron with Sequential Learning """

    def __init__(self, inputs, targets):
        self.nIn = np.shape(inputs)[1]
        self.nOut = np.shape(targets)[1] if np.ndim(targets) > 1 else 1
        self.nData = np.shape(inputs)[0]

        # Initialize weights randomly
        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        """ Train using sequential update """

        # Add bias node to inputs
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)

        for n in range(nIterations):
            for m in range(self.nData):  # Process one input at a time
                inputs_seq = np.array([inputs[m]])  # Single input row
                targets_seq = np.array([targets[m]])  # Corresponding target

                # Forward pass
                self.activations = self.pcnfwd(inputs_seq)

                # Sequential weight update (one sample at a time)
                self.weights -= eta * np.dot(np.transpose(inputs_seq), (self.activations - targets_seq))

    def pcnfwd(self, inputs):
        """ Run forward pass """
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)  # Threshold function

    def confmat(self, inputs, targets):
        """ Compute accuracy """
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = self.pcnfwd(inputs)
        accuracy = np.mean(outputs == targets) * 100
        return accuracy

# === Step 3: Train Perceptron and Evaluate ===
perceptron = PerceptronSequential(inputs, targets)
perceptron.pcntrain(inputs, targets, eta=0.1, nIterations=10)
accuracy = perceptron.confmat(inputs, targets)

# === Step 4: Fix Decision Boundary for Only Two Features ===
# Select only two features for visualization (e.g., "fixed acidity" and "volatile acidity")
feature1, feature2 = 0, 1  # First two features
X_vis = inputs[:, [feature1, feature2]]  # Select first two columns only

# Create mesh grid for plotting
x_min, x_max = X_vis[:, 0].min() - 0.1, X_vis[:, 0].max() + 0.1
y_min, y_max = X_vis[:, 1].min() - 0.1, X_vis[:, 1].max() + 0.1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]  # Only two features used

# Create a dummy input matrix with 11 features, filling unused ones with zero
grid_full = np.zeros((grid.shape[0], 11))  # 11 features
grid_full[:, [feature1, feature2]] = grid  # Assign values to selected features

# Add bias term
grid_full = np.concatenate((grid_full, -np.ones((grid_full.shape[0], 1))), axis=1)

# Forward pass with corrected feature dimensions
Z = perceptron.pcnfwd(grid_full)
Z = Z.reshape(xx.shape)

# === Step 5: Plot Decision Boundary ===
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=targets.ravel(), cmap="coolwarm", edgecolors="k")
plt.xlabel("Fixed Acidity (normalized)")
plt.ylabel("Volatile Acidity (normalized)")
plt.title(f"Perceptron Decision Boundary (Accuracy: {accuracy:.2f}%)")
plt.show()

# Print final accuracy
print(f"Final Accuracy: {accuracy:.2f}%")
