import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# =========================
# CREATE INSURANCE DATASET
# =========================

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "Age": np.random.randint(20, 66, n),
    "Income": np.random.randint(20000, 120001, n),
    "BMI": np.round(np.random.uniform(18, 40, n), 1),
    "Existing_Policy": np.random.choice(["Yes", "No"], n),
    "Previous_Claims": np.random.randint(0, 5, n),
    "Smoker": np.random.choice(["Yes", "No"], n),
    "Medical_Condition": np.random.choice(
        ["None", "Minor", "Major"],
        n,
        p=[0.60, 0.25, 0.15]
    )
})

score = (
    0.03 * (data["Age"] - 40)
    + 0.00001 * (data["Income"] - 60000)
    + 0.08 * (data["BMI"] - 25)
    + 0.30 * data["Existing_Policy"].map({"No": 0, "Yes": 1})
    + 0.20 * data["Previous_Claims"]
    + 0.40 * data["Smoker"].map({"No": 0, "Yes": 1})
    + data["Medical_Condition"].map({
        "None": 0,
        "Minor": 0.25,
        "Major": 0.60
    })
)

probability = 1 / (1 + np.exp(-score))

data["Buy_Insurance"] = np.where(
    np.random.random(n) < probability,
    "Yes",
    "No"
)

print(data.shape)
print(data.head())
y = data["Buy_Insurance"].map({"No": 0, "Yes": 1})

X = data[
    [
        "Age",
        "Income",
        "BMI",
        "Previous_Claims",
        "Existing_Policy",
        "Smoker",
        "Medical_Condition"
    ]
].copy()

# First: 80% train, 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split remaining 20% into 10% validation + 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = [
    "Age",
    "Income",
    "BMI",
    "Previous_Claims"
]

categorical_features = [
    "Existing_Policy",
    "Smoker",
    "Medical_Condition"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape)

import torch

X_train_tensor = torch.tensor(
    X_train_processed,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_processed,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val.values,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_processed,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
)

#Tensor = numerical data structure used by PyTorch
#  for neural-network computation and automatic gradient calculation.

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(11, 16),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(8, 1),
    nn.Sigmoid()
)

print(model)

num_yes = (y_train_tensor == 1).sum()
num_no = (y_train_tensor == 0).sum()

no_weight = num_yes / num_no

print("No class weight:", no_weight.item())

criterion = nn.BCELoss(reduction="none")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

train_losses = []
val_losses = []

epochs = 100

for epoch in range(epochs):

    # -------------------
    # TRAINING
    # -------------------
    model.train()

    train_predictions = model(X_train_tensor).squeeze()

    individual_losses = criterion(
        train_predictions,
        y_train_tensor
    )

    sample_weights = torch.where(
        y_train_tensor == 0,
        no_weight,
        torch.tensor(1.0)
    )

    train_loss = (
        individual_losses * sample_weights
    ).mean()

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()


    # -------------------
    # VALIDATION
    # -------------------
    model.eval()

    with torch.no_grad():

        val_predictions = model(X_val_tensor).squeeze()

        val_individual_losses = criterion(
            val_predictions,
            y_val_tensor
        )

        val_loss = val_individual_losses.mean()

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())


    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss.item():.4f}, "
            f"Val Loss = {val_loss.item():.4f}"
        )


import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.show()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =========================
# TEST SET EVALUATION
# =========================

model.eval()

with torch.no_grad():
    test_probabilities = model(X_test_tensor).squeeze()

# Convert probabilities to classes
test_predictions = (test_probabilities >= 0.5).int()

# Convert tensors to NumPy
y_test_numpy = y_test_tensor.numpy()
test_predictions_numpy = test_predictions.numpy()
test_probabilities_numpy = test_probabilities.numpy()

# Metrics
accuracy = accuracy_score(
    y_test_numpy,
    test_predictions_numpy
)

precision = precision_score(
    y_test_numpy,
    test_predictions_numpy
)

recall = recall_score(
    y_test_numpy,
    test_predictions_numpy
)

f1 = f1_score(
    y_test_numpy,
    test_predictions_numpy
)

roc_auc = roc_auc_score(
    y_test_numpy,
    test_probabilities_numpy
)

print("\n===== TEST RESULTS =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test_numpy,
    test_predictions_numpy
))

print("\nClassification Report:")
print(
    classification_report(
        y_test_numpy,
        test_predictions_numpy
    )
)

print("\nFirst 20 test probabilities:")

for probability, actual in zip(
    test_probabilities[:20],
    y_test_tensor[:20]
):
    print(
        f"Probability={probability.item():.3f}, "
        f"Actual={int(actual.item())}"
    )

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

    predictions = (
        test_probabilities_numpy >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test_numpy,
        predictions
    )

    precision = precision_score(
        y_test_numpy,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_numpy,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_numpy,
        predictions,
        zero_division=0
    )

    print(
        f"Threshold={threshold:.1f} | "
        f"Accuracy={accuracy:.3f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

# we identified class imbalance so solcing it
print("Training class distribution:")
print(y_train.value_counts())

# Recall = 0.929 → we still correctly identify most actual Yes customers.
# Precision = 0.815 → we're also getting fewer false Yes predictions.
# So some customers who were previously classified as Yes are now classified as No.

