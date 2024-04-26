import numpy as np
import time


def get_mnist(classes=None):
    data = np.genfromtxt("D:/vs code/PROJECTS/archive/mnist_train.csv", delimiter=",", skip_header=1)
    images = data[:, 1:]  # Extract pixel values
    labels = data[:, 0]   # Extract labels
    images = images.astype("float32") / 255
    
    if classes is not None:
        indices = np.where(np.isin(labels, classes))
        images = images[indices]
        labels = labels[indices]
    
    labels = np.eye(10)[labels.astype(int)]
    return images, labels

def get_mnist_test(classes=None):
    data = np.genfromtxt("D:/vs code/PROJECTS/archive/mnist_test.csv", delimiter=",", skip_header=1)
    images = data[:, 1:]  # Extract pixel values
    labels = data[:, 0]   # Extract labels
    images = images.astype("float32") / 255

    if classes is not None:
        indices = np.where(np.isin(labels, classes))
        images = images[indices]
        labels = labels[indices]

    labels = np.eye(10)[labels.astype(int)]
    return images, labels

def ask_user_classes():
    choice = input("Do you want to train for all 10 classes? (yes/no): ").lower()
    if choice == 'yes':
        return None
    elif choice == 'no':
        classes = list(map(int, input("Enter the classes you want to train separated by space: ").split()))
        return classes
    else:
        print("Invalid choice. Defaulting to training for all 10 classes.")
        return None

def calculate_learning_rate(iteration):
    if iteration > 0:
        return 0.01
    return 1 / np.sqrt(iteration + 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backward_propagation(X, y, z1, a1, z2, a2, W2):
    delta2 = (a2 - y) * sigmoid_prime(z2)
    grad_W2 = np.dot(a1.T, delta2)
    grad_b2 = np.sum(delta2, axis=0)

    delta1 = np.dot(delta2, W2.T) * sigmoid_prime(z1)
    grad_W1 = np.dot(X.T, delta1)
    grad_b1 = np.sum(delta1, axis=0)

    return grad_W1, grad_b1, grad_W2, grad_b2

def train(X_train, y_train, epochs, batch_size):
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = 10

    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)

    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        total_correct = 0
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size

            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            z1, a1, z2, a2 = forward_propagation(X_batch, W1, b1, W2, b2)
            predictions = np.argmax(a2, axis=1)
            total_correct += np.sum(predictions == np.argmax(y_batch, axis=1))

            # Backpropagation
            delta2 = (a2 - y_batch)
            grad_W2 = np.dot(a1.T, delta2)
            grad_b2 = np.sum(delta2, axis=0)

            delta1 = np.dot(delta2, W2.T) * sigmoid_prime(z1)
            grad_W1 = np.dot(X_batch.T, delta1)
            grad_b1 = np.sum(delta1, axis=0)

            t = epoch * num_batches + batch + 1
            learning_rate = calculate_learning_rate(t)

            # Update weights and biases
            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

        accuracy = (total_correct / len(X_train)) * 100
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}%")

    return W1, b1, W2, b2

def test(X_test, y_test, W1, b1, W2, b2):
    _, _, _, a2 = forward_propagation(X_test, W1, b1, W2, b2)
    predictions = np.argmax(a2, axis=1)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

def main():
    # Training
    classes = ask_user_classes()
    X_train, y_train = get_mnist(classes)
    epochs = 10
    batch_size = 64
    start_time = time.time()
    print("Training...")
    W1, b1, W2, b2 = train(X_train, y_train, epochs, batch_size)
    end_time = time.time()  # End time tracking
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")


    # Testing
    X_test, y_test = get_mnist_test(classes)
    print("Testing...")
    test(X_test, y_test, W1, b1, W2, b2)

if __name__ == "__main__":
    main()
