from core.DataFrames import create_polars_dataframe_by_subplot, feature_cols
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from imblearn.over_sampling import SMOTE


def main():
    # --- Setup ---
    # Feature columns to use from the dataset
    selected_feature_cols = [
        "LIVE_BA",
        "LIVE10to30",
        "TPA"
    ]
    dependent_var_col = "OFE"

    # --- Data Loading and Preparation ---
    print("Loading data...")
    # Load the OFE data
    ofe_data = pd.read_csv("data/subplot-ofe-AZ.csv")
    print(f"OFE data loaded with {len(ofe_data)} rows")

    # Check if we have enough data
    if len(ofe_data) == 0:
        print("Error: No matching data found between forest plot data and OFE data")
        return
    
    # Split data into train and test sets
    X = ofe_data[selected_feature_cols]
    y = ofe_data[dependent_var_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    print(f"OFE class distribution in training data: \n{y_train.value_counts()}")

    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("SMOTE applied. New class distribution in training data:")
    print(y_train_resampled.value_counts())

    # --- Model Training ---
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    # --- Model Evaluation ---
    print("Evaluating model...")


    # 1. Get the predicted probabilities for the positive class (OGF)
    # model.predict_proba(X_test) returns an array like [[prob_0, prob_1], [prob_0, prob_1], ...]
    # We only care about the probability of class 1.
    predicted_probabilities = model.predict_proba(X_test)[:, 1]

    # 2. Set your custom threshold
    # Start with something higher than 0.5 and experiment. Let's try 0.7.
    CUSTOM_THRESHOLD = 0.8

    # 3. Apply the threshold to get the new predictions
    y_pred_custom = (predicted_probabilities >= CUSTOM_THRESHOLD).astype(int)

    print(f"\n--- Evaluation with Custom Threshold of {CUSTOM_THRESHOLD} ---")

    # 4. Evaluate using the new custom predictions
    print("Classification Report:")
    print(classification_report(y_test, y_pred_custom))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_custom))

    # You can keep the original evaluation to compare
    print("\n--- Original Evaluation (0.5 Threshold) ---")
    y_pred_original = model.predict(X_test)
    print(classification_report(y_test, y_pred_original))

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': selected_feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    # Save the model
    if not os.path.exists('weights'):
        os.makedirs('weights')
    joblib.dump(model, 'weights/random_forest_ofe_model.joblib')
    print("Model saved to weights/random_forest_ofe_model.joblib")

if __name__ == "__main__":
    main()