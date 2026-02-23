# train_model.py

from sklearn.ensemble import RandomForestClassifier
import pickle

# Example data
X = [[3.5, 8], [3.8, 7], [2.5, 5]]  # GPA, Python skill
y = ["Data Scientist", "Data Analyst", "Web Developer"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")