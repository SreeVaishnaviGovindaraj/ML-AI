import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Age": np.random.randint(20, 66, n),
    "Income": np.random.randint(20000, 100001, n),
    "Existing_Policy": np.random.choice(["Yes", "No"], n),
    "Previous_Claims": np.random.randint(0, 4, n)
})

# Create the target (whether customer buys insurance)
data["Buy_Insurance"] = np.where(
    (
        (data["Age"] > 35) &
        (data["Income"] > 50000)
    ) |
    (data["Existing_Policy"] == "Yes") |
    (data["Previous_Claims"] > 1),
    "Yes",
    "No"
)

print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data["Buy_Insurance"].value_counts())
data.to_csv("insurance_customers.csv", index=False)


# ____________

from sklearn.model_selection import train_test_split

X = data[
    ["Age", "Income", "Existing_Policy", "Previous_Claims"]
]

y = data["Buy_Insurance"].map({
    "No": 0,
    "Yes": 1
})

# Convert categorical feature to numbers
X["Existing_Policy"] = X["Existing_Policy"].map({
    "No": 0,
    "Yes": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
print(y_test)

#logistic regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train) # get the X_train>features, y_train>known answers and learns the relationship between them
y_pred = model.predict(X_test)
print("Actual:")
print(y_test.head(10).values)

print("Predicted:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
## [[ TN  FP ] [17 14]
## [ FN  TP ]] [3 166]
# Accuracy	91.5% -(166 + 17) / 200
# Precision	92.2% -166 / (166 + 14)
# Recall	98.2% -166 / (166 + 3)
# F1	95.1%

y_prob = model.predict_proba(X_test)[:, 1]

print(y_prob[:10])
threshold = 0.5

y_pred_05 = (y_prob >= threshold).astype(int)

print(y_pred_05[:10])
threshold = 0.3

y_pred_03 = (y_prob >= threshold).astype(int)

print(y_pred_05[:10])
threshold = 0.7

y_pred_03 = (y_prob >= threshold).astype(int)

print(y_pred_05[:10])
#____________scaling the features_____________
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(y_pred[:10])
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
#_____________Regularization______________-
model = LogisticRegression(C=0.01) #0.01 very strong , 0.1 strong, 1.0 moderate, 10 weak, 100 very weak 
#the model is given a much stronger incentive to keep those weights small
#0.1 is the default
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:10])
print("Weights:", model.coef_) #learned weights are stored
print("Bias:", model.intercept_) #bias/intercept
#________cross validation____________
from sklearn.model_selection import cross_val_score

model = LogisticRegression(C=1.0)

scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("CV scores:", scores)
print("Mean CV accuracy:", scores.mean())

#________Hyperparameter tunning______________
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)

print("Best C:", grid.best_params_)
print("Best CV score:", grid.best_score_)