import numpy as np

class pcn:
    """ A basic Perceptron with Sequential Learning """

    def __init__(self, inputs, targets):
        """ Constructor """
        self.nIn = np.shape(inputs)[1] if np.ndim(inputs) > 1 else 1
        self.nOut = np.shape(targets)[1] if np.ndim(targets) > 1 else 1
        self.nData = np.shape(inputs)[0]

        # Initialize network weights randomly
        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        """ Train using sequential update """

        # Add bias node to inputs
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)

        print("\nInitial weights:")
        print(self.weights)

        for n in range(nIterations):
            print(f"\nIteration {n + 1}:")

            for m in range(self.nData):  # Process one input at a time
                inputs_seq = np.array([inputs[m]])  # Single input row
                targets_seq = np.array([targets[m]])  # Corresponding target

                # Forward pass
                self.activations = self.pcnfwd(inputs_seq)

                # Sequential weight update (one sample at a time)
                self.weights -= eta * np.dot(np.transpose(inputs_seq), (self.activations - targets_seq))

                print(f"Step {m + 1}, Updated weights:")
                print(self.weights)

    def pcnfwd(self, inputs):
        """ Run forward pass """
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)  # Threshold function

    def confmat(self, inputs, targets):
        """ Compute confusion matrix """
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = self.pcnfwd(inputs)
        print("\nFinal Outputs:")
        print(outputs)

# === TEST CASE: AND Gate ===
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])  # AND gate inputs
targets = np.array([[0], [0], [0], [1]])  # AND gate outputs

# Create Perceptron instance and train
perceptron = pcn(inputs, targets)
perceptron.pcntrain(inputs, targets, eta=0.1, nIterations=10)
perceptron.confmat(inputs, targets)
