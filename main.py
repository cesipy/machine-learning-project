
import pandas as pd
import numpy as np
import os
import pathlib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    train_data = pd.read_csv("transactions.csv.zip")
    X = train_data.drop(columns = "Class")
    Y = train_data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Model selection
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

# Training and evaluation
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"{name} CV Accuracy: {np.mean(cv_scores)}")

        # Fit the model
        model.fit(X_train_scaled, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test_scaled)
        
        # Evaluation metrics
        print(f"{name} Test Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

