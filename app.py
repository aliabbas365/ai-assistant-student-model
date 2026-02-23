# ==============================
# CS Students ML Model
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("cs_students.csv")

print("Dataset Loaded")
print(df.head())

# ------------------------------
# 2. Drop Unnecessary Columns
# ------------------------------
df = df.drop(columns=["Student ID", "Name"], errors="ignore")

# ------------------------------
# 3. Check Data
# ------------------------------
print("\nDataset Info:")
print(df.info())

# ------------------------------
# 4. Convert Categorical Data
# ------------------------------
categorical_cols = ["Gender", "Major", "Interested Domain", "Projects", "Future Career", "SQL", "Java"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nAfter Encoding:")
print(df.head())

# ------------------------------
# 5. Select Target
# ------------------------------
# We will predict Python skill
target_column = "Python"

X = df.drop(columns=[target_column])
y = df[target_column]

# ------------------------------
# 6. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# ------------------------------
# 7. Train Model
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ------------------------------
# 8. Predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# 9. Evaluation
# ------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------
# 10. Feature Importance
# ------------------------------
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Important Features:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10,5))
plt.bar(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xticks(rotation=45)
plt.title("Top Features Affecting Python Skill")
plt.show()

