# Define the inputs and target
import numpy as np

# Inputs
x = np.array([2.0, 3.0])

# Target
y = 1.0

#Initialize weights and biases
# Hidden layer
w1 = np.array([
    [0.5, 0.3],   # weights connected to x1
    [0.4, 0.6]    # weights connected to x2
])

b1 = np.array([0.1, -0.2])

# Output layer
w2 = np.array([0.7, 0.8])
b2 = -1.0
#Activation functions
def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Forward propagation
# Hidden layer
z1 = np.dot(x, w1) + b1

# ReLU activation
h = relu(z1)

# Output layer
z2 = np.dot(h, w2) + b2

# Sigmoid
p = sigmoid(z2)

print("Hidden pre-activation:", z1)
print("Hidden output:", h)
print("Output pre-activation:", z2)
print("Predicted probability:", p)
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

print("Loss:", loss)
d_z2 = p - y

print("Output gradient:", d_z2)
d_w2 = d_z2 * h
d_b2 = d_z2

print("Gradient w2:", d_w2)
print("Gradient b2:", d_b2)
learning_rate = 0.1
w2 = w2 - learning_rate * d_w2
b2 = b2 - learning_rate * d_b2

print("Updated w2:", w2)
print("Updated b2:", b2)
d_h = d_z2 * w2
d_z1 = d_h * (z1 > 0)
d_w1 = np.outer(x, d_z1)
d_b1 = d_z1

print("Gradient w1:")
print(d_w1)

print("Gradient b1:")
print(d_b1)
w1 = w1 - learning_rate * d_w1
b1 = b1 - learning_rate * d_b1

print("Updated w1:")
print(w1)

print("Updated b1:")
print(b1)
for epoch in range(1000):

    # Forward
    z1 = np.dot(x, w1) + b1
    h = relu(z1)

    z2 = np.dot(h, w2) + b2
    p = sigmoid(z2)

    # Loss
    loss = -(y * np.log(p) + (1-y) * np.log(1-p))

    # Backpropagation
    d_z2 = p - y

    d_w2 = d_z2 * h
    d_b2 = d_z2

    d_h = d_z2 * w2
    d_z1 = d_h * (z1 > 0)

    d_w1 = np.outer(x, d_z1)
    d_b1 = d_z1

    # Update
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2

    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1

    if epoch % 100 == 0:
        print(epoch, loss)