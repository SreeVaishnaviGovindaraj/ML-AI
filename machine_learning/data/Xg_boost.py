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

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

xgb_model.fit(
    X_train_tree,
    y_train_tree
)

y_pred_xgb = xgb_model.predict(X_test_tree)

print("Train Accuracy:", xgb_model.score(
    X_train_tree,
    y_train_tree
))

print("Test Accuracy:", xgb_model.score(
    X_test_tree,
    y_test_tree
))


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.01,
    max_depth=3,
    random_state=42
)

xgb_model.fit(X_train_tree, y_train_tree)

print("Train Accuracy:", xgb_model.score(X_train_tree, y_train_tree))
print("Test Accuracy:", xgb_model.score(X_test_tree, y_test_tree))