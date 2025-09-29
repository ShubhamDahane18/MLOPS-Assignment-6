# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
