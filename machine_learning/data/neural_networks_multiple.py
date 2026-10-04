#Create a small dataset
import numpy as np

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
], dtype=float)

y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

print(X)
print(y)
#Initialize the network
np.random.seed(42)

# Input → Hidden
W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros(2)

# Hidden → Output
W2 = np.random.randn(2) * 0.1
b2 = 0.0

learning_rate = 0.1
#Forward propagation
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Z1 = X @ W1 + b1
H = relu(Z1)

Z2 = H @ W2 + b2
P = sigmoid(Z2)

print("Predictions:")
print(P)
# loss = -(y * np.log(p) + (1-y) * np.log(1-p))
loss = -np.mean(
    y * np.log(P) +
    (1 - y) * np.log(1 - P)
)

print("Loss:", loss)
dZ2 = P - y
m = len(X) #no of training samples

dZ2 = (P - y) / m

# Output-layer gradients
dW2 = H.T @ dZ2
# H.T @ dZ2
#      ↓
# combines gradients from all 6 students
#      ↓
# produces gradients for W2
db2 = np.sum(dZ2)
dH = dZ2[:, None] * W2

dZ1 = dH * (Z1 > 0) #(Z1 > 0) is the derivative of ReLU.
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0)
W2 -= learning_rate * dW2
b2 -= learning_rate * db2

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

for epoch in range(1000):

    # 1. Forward propagation
    Z1 = X @ W1 + b1
    H = relu(Z1)

    Z2 = H @ W2 + b2
    P = sigmoid(Z2)

    # 2. Loss
    m = len(X)

    loss = -np.mean(
        y * np.log(P) +
        (1 - y) * np.log(1 - P)
    )

    # 3. Backpropagation
    dZ2 = (P - y) / m

    dW2 = H.T @ dZ2
    db2 = np.sum(dZ2)

    dH = dZ2[:, None] * W2
    dZ1 = dH * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    # 4. Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # 5. Display loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Too few epochs:
# Under-trained → model hasn't learned enough

# Too many epochs:
# Potential overfitting → model starts memorizing training data

# So the practical approach is:
# Choose maximum epochs
#         ↓
# Train
#         ↓
# Monitor training/validation loss
#         ↓
# Stop when improvement stops

#Make predictions
# Forward pass using the final trained weights

Z1 = X @ W1 + b1
H = relu(Z1)

Z2 = H @ W2 + b2
P = sigmoid(Z2)

print("Probabilities:")
print(P)

#Convert probability to class

predictions = (P >= 0.5).astype(int)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)

np.random.seed(42)

X = np.array([
    [1,2], [1,3], [2,2], [2,3], [2,4],
    [3,3], [3,4], [3,5], [4,4], [4,5],
    [4,6], [5,5], [5,6], [5,7], [6,6],
    [6,7], [6,8], [7,7], [7,8], [8,8]
], dtype=float)

y = np.array([
    0,0,0,0,0,
    0,0,1,0,1,
    1,1,1,1,1,
    1,1,1,1,1
], dtype=float)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

for epoch in range(1000):

    Z1 = X_train @ W1 + b1
    H = relu(Z1)

    Z2 = H @ W2 + b2
    P = sigmoid(Z2)

    m = len(X_train)

    loss = -np.mean(
        y_train * np.log(P) +
        (1 - y_train) * np.log(1 - P)
    )

    dZ2 = (P - y_train) / m

    dW2 = H.T @ dZ2
    db2 = np.sum(dZ2)

    dH = dZ2[:, None] * W2
    dZ1 = dH * (Z1 > 0)

    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
# Test the trained neural network

Z1_test = X_test @ W1 + b1
H_test = relu(Z1_test)

Z2_test = H_test @ W2 + b2
P_test = sigmoid(Z2_test)

print("Test probabilities:")
print(P_test)
predictions = (P_test >= 0.5).astype(int)

print("Predictions:")
print(predictions)

print("Actual:")
print(y_test)

accuracy = np.mean(predictions == y_test)

print("Test Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# _________________________________
# Pytorch version
import torch

print(torch.__version__)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

print(model)

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
for epoch in range(1000):

    # Forward pass
    predictions = model(X_train_tensor)

    # Loss
    loss = criterion(
        predictions.squeeze(),
        y_train_tensor
    )

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    test_probabilities = model(X_test_tensor).squeeze()

print("Test probabilities:")
print(test_probabilities)
test_predictions = (test_probabilities >= 0.5).int()

print("Predictions:")
print(test_predictions)

print("Actual:")
print(torch.tensor(y_test, dtype=torch.int32))
x = torch.tensor(2.0, requires_grad=True)

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# -----------------------------------
# AUTOGRAD DEMONSTRATION
# -----------------------------------

# Check parameters BEFORE backpropagation
print("\nBEFORE BACKPROPAGATION")

for name, param in model.named_parameters():
    print(name)
    print("Gradient:", param.grad)


# One forward pass
predictions = model(X_train_tensor)

# Calculate loss
loss = criterion(
    predictions.squeeze(),
    y_train_tensor
)

print("\nLoss:", loss.item())

# Clear previous gradients
optimizer.zero_grad()

# Calculate gradients
loss.backward()

print("\nAFTER BACKPROPAGATION")

for name, param in model.named_parameters():
    print(name)
    print("Gradient:")
    print(param.grad)

optimizer.step()

print("\nAFTER WEIGHT UPDATE")

for name, param in model.named_parameters():
    print(name)
    print(param)

optimizer.zero_grad()

print("BEFORE:")
for name, param in model.named_parameters():
    print(name, param.grad)

predictions = model(X_train_tensor)

loss = criterion(
    predictions.squeeze(),
    y_train_tensor
)

loss.backward()

print("AFTER:")
for name, param in model.named_parameters():
    print(name, param.grad)