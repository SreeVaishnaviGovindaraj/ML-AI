# ============================================================
# DECISION TREE - DATA CREATION
# ============================================================

import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

n = 1000

data_tree = pd.DataFrame({
    "Age": np.random.randint(20, 66, n),
    "Income": np.random.randint(20000, 100001, n),
    "Existing_Policy": np.random.choice(["Yes", "No"], n),
    "Previous_Claims": np.random.randint(0, 4, n)
})

# Create target
data_tree["Buy_Insurance"] = np.where(
    (
        (data_tree["Age"] > 35) &
        (data_tree["Income"] > 50000)
    ) |
    (data_tree["Existing_Policy"] == "Yes") |
    (data_tree["Previous_Claims"] > 1),
    "Yes",
    "No"
)

print(data_tree.head())
print(data_tree.shape)
print(data_tree["Buy_Insurance"].value_counts())

# Features
X_tree = data_tree[
    ["Age", "Income", "Existing_Policy", "Previous_Claims"]
].copy()

# Target
y_tree = data_tree["Buy_Insurance"].map({
    "No": 0,
    "Yes": 1
})

# Encode categorical feature
X_tree["Existing_Policy"] = X_tree["Existing_Policy"].map({
    "No": 0,
    "Yes": 1
})

from sklearn.model_selection import train_test_split

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_tree,
    y_tree,
    test_size=0.2,
    random_state=42
)

print("Training:", X_train_tree.shape)
print("Testing:", X_test_tree.shape)

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_train_tree,
    y_train_tree
)

y_pred_tree = tree_model.predict(X_test_tree)

print("Actual:")
print(y_test_tree.head(10).values)

print("Predicted:")
print(y_pred_tree[:10])

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(
    y_test_tree,
    y_pred_tree
)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test_tree,
    y_pred_tree
))

print("\nClassification Report:")
print(classification_report(
    y_test_tree,
    y_pred_tree
))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=X_train_tree.columns,
    class_names=["No", "Yes"],
    filled=True
)

plt.show()

for depth in [2, 3, 5, 10, None]:
    tree_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(X_train_tree, y_train_tree)

    train_accuracy = tree_model.score(X_train_tree, y_train_tree)
    test_accuracy = tree_model.score(X_test_tree, y_test_tree)

    print(
        "Depth:", depth,
        "| Train:", train_accuracy,
        "| Test:", test_accuracy
    )
from sklearn.tree import DecisionTreeClassifier

for split in [2, 10, 20, 50, 100]:

    tree_model = DecisionTreeClassifier(
        min_samples_split=split,
        random_state=42
    )

    tree_model.fit(
        X_train_tree,
        y_train_tree
    )

    train_accuracy = tree_model.score(
        X_train_tree,
        y_train_tree
    )

    test_accuracy = tree_model.score(
        X_test_tree,
        y_test_tree
    )

    print(
        "min_samples_split:", split,
        "| Train:", train_accuracy,
        "| Test:", test_accuracy
    )

from sklearn.tree import DecisionTreeClassifier

for leaf in [1, 5, 10, 20, 50, 100]:

    tree_model = DecisionTreeClassifier(
        min_samples_leaf=leaf,
        random_state=42
    )

    tree_model.fit(
        X_train_tree,
        y_train_tree
    )

    train_accuracy = tree_model.score(
        X_train_tree,
        y_train_tree
    )

    test_accuracy = tree_model.score(
        X_test_tree,
        y_test_tree
    )

    print(
        "min_samples_leaf:", leaf,
        "| Train:", train_accuracy,
        "| Test:", test_accuracy
    )

from sklearn.tree import DecisionTreeClassifier

for leaves in [2, 5, 10, 20, 50, 100]:

    tree_model = DecisionTreeClassifier(
        max_leaf_nodes=leaves,
        random_state=42
    )

    tree_model.fit(
        X_train_tree,
        y_train_tree
    )

    train_accuracy = tree_model.score(
        X_train_tree,
        y_train_tree
    )

    test_accuracy = tree_model.score(
        X_test_tree,
        y_test_tree
    )

    print(
        "max_leaf_nodes:", leaves,
        "| Train:", train_accuracy,
        "| Test:", test_accuracy
    )

