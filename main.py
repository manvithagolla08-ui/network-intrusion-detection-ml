
import pandas as pd
import lightgbm as lgb

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Separate features and target
X = train.drop(columns=["Target", "id"])
y = train["Target"]

X_test = test.drop(columns=["id"])

# Handle categorical columns
categorical_cols = ["proto", "service"]

for col in categorical_cols:
    categories = pd.Index(
        pd.concat([X[col], X_test[col]]).astype(str).unique()
    )
    X[col] = pd.Categorical(X[col].astype(str), categories=categories)
    X_test[col] = pd.Categorical(
        X_test[col].astype(str), categories=categories
    )

# Original LightGBM model
model1 = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=137,
    learning_rate=0.07,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# Regularized LightGBM model
model2 = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=7,
    reg_lambda=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# Train both models
model1.fit(X, y, categorical_feature=categorical_cols)
model2.fit(X, y, categorical_feature=categorical_cols)

# Blend predictions
prob1 = model1.predict_proba(X_test)
prob2 = model2.predict_proba(X_test)

blended_prob = 0.8 * prob1 + 0.2 * prob2

predictions = model1.classes_[blended_prob.argmax(axis=1)]

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "Target": predictions
})

submission.to_csv("generated_submission.csv", index=False)

print("Prediction file created successfully!")
