import numpy as np
import pandas as pd
# Define helper functions
def conv2d(X, W, b, stride=1, padding=0):
    """
    Perform a 2D convolution operation.
    """
    n_samples, in_channels, in_height, in_width = X.shape
    n_filters, _, f_height, f_width = W.shape

    # Add padding to the input
    X_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

    # Calculate output dimensions
    out_height = (in_height - f_height + 2 * padding) // stride + 1
    out_width = (in_width - f_width + 2 * padding) // stride + 1

    # Initialize the output
    out = np.zeros((n_samples, n_filters, out_height, out_width))

    for i in range(n_samples):
        for j in range(n_filters):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + f_height
                    w_end = w_start + f_width

                    X_slice = X_padded[i, :, h_start:h_end, w_start:w_end]
                    out[i, j, h, w] = np.sum(X_slice * W[j]) + b[j]

    return out

def relu(X):
    return np.maximum(0, X)

def max_pooling(X, size=2, stride=2):
    """
    Perform max pooling operation.
    """
    n_samples, n_channels, in_height, in_width = X.shape

    out_height = (in_height - size) // stride + 1
    out_width = (in_width - size) // stride + 1

    out = np.zeros((n_samples, n_channels, out_height, out_width))

    for i in range(n_samples):
        for c in range(n_channels):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + size
                    w_end = w_start + size

                    X_slice = X[i, c, h_start:h_end, w_start:w_end]
                    out[i, c, h, w] = np.max(X_slice)

    return out

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Define the CNN class
class SimpleCNN:
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        # Initialize weights and biases
        self.W1 = np.random.randn(8, input_shape[0], 3, 3) * 0.1
        self.b1 = np.zeros((8, 1))
        self.W2 = np.random.randn(16, 8, 3, 3) * 0.1
        self.b2 = np.zeros((16, 1))
        self.W3 = np.random.randn(16 * 5 * 5, 128) * 0.1
        self.b3 = np.zeros((128,))
        self.W4 = np.random.randn(128, num_classes) * 0.1
        self.b4 = np.zeros((num_classes,))

    def forward(self, X):
        # Forward propagation
        self.Z1 = conv2d(X, self.W1, self.b1, stride=1, padding=1)
        self.A1 = relu(self.Z1)
        self.P1 = max_pooling(self.A1, size=2, stride=2)

        self.Z2 = conv2d(self.P1, self.W2, self.b2, stride=1, padding=1)
        self.A2 = relu(self.Z2)
        self.P2 = max_pooling(self.A2, size=2, stride=2)

        self.P2_flat = self.P2.reshape(X.shape[0], -1)
        self.Z3 = np.dot(self.P2_flat, self.W3) + self.b3
        self.A3 = relu(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = softmax(self.Z4)

        return self.A4

    def compute_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
        return loss

    def backward(self, X, Y_true):
        m = X.shape[0]
        dZ4 = self.A4 - Y_true  # Shape: (m, num_classes)
        dW4 = np.dot(self.A3.T, dZ4) / m
        db4 = np.sum(dZ4, axis=0) / m

        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * (self.Z3 > 0)
        dW3 = np.dot(self.P2_flat.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0) / m

        dP2_flat = np.dot(dZ3, self.W3.T)
        dP2 = dP2_flat.reshape(self.P2.shape)

        # Backprop through max pooling layer 2
        dA2 = self.max_pooling_backward(dP2, self.A2, self.P2)

        # Backprop through convolutional layer 2
        dZ2 = dA2 * (self.Z2 > 0)
        dW2, db2, dP1 = self.conv_backward(dZ2, self.A1, self.W2, padding=1)

        # Backprop through max pooling layer 1
        dA1 = self.max_pooling_backward(dP1, self.A1, self.P1)

        # Backprop through convolutional layer 1
        dZ1 = dA1 * (self.Z1 > 0)
        dW1, db1, _ = self.conv_backward(dZ1, X, self.W1, padding=1)

        # Update gradients
        self.grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3,
            'dW4': dW4, 'db4': db4
        }

    def conv_backward(self, dZ, A_prev, W, padding=0):
        n_samples, n_filters, out_height, out_width = dZ.shape
        _, _, f_height, f_width = W.shape
        dW = np.zeros_like(W)
        db = np.zeros_like(W[:, 0, 0, 0])
        dA_prev = np.zeros_like(A_prev)

        A_prev_padded = np.pad(A_prev, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        dA_prev_padded = np.pad(dA_prev, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

        for i in range(n_samples):
            a_prev_padded = A_prev_padded[i]
            da_prev_padded = dA_prev_padded[i]
            for c in range(n_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h
                        h_end = h_start + f_height
                        w_start = w
                        w_end = w_start + f_width

                        a_slice = a_prev_padded[:, h_start:h_end, w_start:w_end]
                        da_prev_padded[:, h_start:h_end, w_start:w_end] += W[c] * dZ[i, c, h, w]
                        dW[c] += a_slice * dZ[i, c, h, w]
                        db[c] += dZ[i, c, h, w]

            if padding == 0:
                dA_prev[i, :, :, :] = da_prev_padded
            else:
                dA_prev[i, :, :, :] = da_prev_padded[:, padding:-padding, padding:-padding]

        dW /= n_samples
        db = db.reshape(-1, 1) / n_samples
        return dW, db, dA_prev

    def max_pooling_backward(self, dP, A_prev, P, size=2, stride=2):
        m, n_channels, out_height, out_width = dP.shape
        dA = np.zeros_like(A_prev)

        for i in range(m):
            for c in range(n_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride
                        h_end = h_start + size
                        w_start = w * stride
                        w_end = w_start + size

                        a_slice = A_prev[i, c, h_start:h_end, w_start:w_end]
                        mask = (a_slice == np.max(a_slice))
                        dA[i, c, h_start:h_end, w_start:w_end] += mask * dP[i, c, h, w]

        return dA

    def update_parameters(self, learning_rate):
        # Update weights and biases
        self.W1 -= learning_rate * self.grads['dW1']
        self.b1 -= learning_rate * self.grads['db1']
        self.W2 -= learning_rate * self.grads['dW2']
        self.b2 -= learning_rate * self.grads['db2']
        self.W3 -= learning_rate * self.grads['dW3']
        self.b3 -= learning_rate * self.grads['db3']
        self.W4 -= learning_rate * self.grads['dW4']
        self.b4 -= learning_rate * self.grads['db4']

    def train(self, X_train, y_train, epochs=10, learning_rate=0.01, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                output = self.forward(X_batch)
                loss = self.compute_loss(y_batch, output)
                self.backward(X_batch, y_batch)
                self.update_parameters(learning_rate)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

if __name__ == "__main__":
    # Load and preprocess data
    # For example, using MNIST dataset
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.to_numpy().reshape(-1, 1, 28, 28) / 255.0
    y = y.astype(int)
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1

    # Split data
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y_one_hot[:60000], y[60000:]

    # Instantiate the CNN
    cnn = SimpleCNN()

    # Train the CNN
    cnn.train(X_train, y_train, epochs=5, learning_rate=0.01, batch_size=64)

    # Evaluate the model
    predictions = cnn.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")