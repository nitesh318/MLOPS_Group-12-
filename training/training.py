import logging
import os
import pickle

import pandas as pd
import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Create the directory for saving models if it doesn't exist
os.makedirs('pkl', exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Load dataset (Iris dataset)
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Data Preprocessing
    df.drop_duplicates(inplace=True)
    X = df.drop(columns=['target'])
    y = df['target']

    # Stratified split for maintaining class proportions
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 40, 60],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
    grid_rf.fit(X_train_scaled, y_train)

    # Save the best model and scaler
    best_model = grid_rf.best_estimator_
    with open('pkl/model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    with open('pkl/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Predictions and evaluation
    predictions = best_model.predict(X_test_scaled)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled), multi_class='ovr')

    # Log evaluation metrics
    logging.info("Classification Report:\n%s", report)
    logging.info("Confusion Matrix:\n%s", cm)
    logging.info("Precision: %f", precision)
    logging.info("Recall: %f", recall)
    logging.info("F1-score: %f", f1)
    logging.info("ROC AUC: %f", roc_auc)

    # SHAP Values Calculation
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    logging.info("SHAP values calculated.")

except Exception as e:
    logging.error("An error occurred: %s", str(e))
