import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_and_evaluate_model(X_train, X_val, y_train, y_val, output_dir="model", model_type='random_forest'):
    """
    Trains a machine learning model and evaluates it on a validation dataset.
    Saves the trained model as a .pkl file in the specified output directory.
    
    Parameters:
        X_train (array-like): Training feature matrix.
        X_val (array-like): Validation feature matrix.
        y_train (array-like): Training labels.
        y_val (array-like): Validation labels.
        output_dir (str): Directory to save the trained model.
        model_type (str): Type of model to train ('random_forest' or 'logistic_regression').
    
    Returns:
        model: The trained machine learning model.
        le: Label encoder used for encoding class labels.
        scaler: Standard scaler used for feature normalization.
    """
    # Convert categorical labels to numerical format
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    # Standardize the feature matrices
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    if model_type == 'logistic_regression':
        # Train a Logistic Regression model
        model = LogisticRegression(
            random_state=42,
            max_iter=500,
            class_weight='balanced',
            solver='lbfgs',  # lbfgs solver supports parallel computation
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    elif model_type == 'random_forest':
        # Define a base RandomForest model
        base_rf = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 300],  # Number of trees in the forest
            'max_depth': [10, 15],       # Maximum depth of trees
            'min_samples_split': [1, 3]  # Minimum samples required to split a node
        }
        grid_search = GridSearchCV(
            base_rf,
            param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters from GridSearchCV:", grid_search.best_params_)
        best_rf = grid_search.best_estimator_

        # Calibrate probability estimates to improve prediction confidence
        model = CalibratedClassifierCV(best_rf, method='sigmoid', cv=3)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model_type. Choose from 'logistic_regression', 'random_forest'.")

    # Make predictions on validation set
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'heart_disease_model_{model_type}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model, le, scaler

if __name__ == "__main__":
    output_directory = "../dataset/processed_dataset"
    model_directory = "model"
    
    # Load dataset
    X_train = np.load(os.path.join(output_directory, 'X_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(output_directory, 'X_val.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(output_directory, 'y_train.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(output_directory, 'y_val.npy'), allow_pickle=True)

    print("Unique values in y_train:", np.unique(y_train))
    print("Unique values in y_val:", np.unique(y_val))

    # Train and evaluate the RandomForest model with hyperparameter tuning
    model, le, scaler = train_and_evaluate_model(
        X_train, X_val, y_train, y_val,
        output_dir=model_directory,
        model_type='random_forest'
    )

    # Load and preprocess the test dataset
    X_test = np.load(os.path.join(output_directory, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(output_directory, 'y_test.npy'), allow_pickle=True)
    X_test = scaler.transform(X_test)
    y_test = le.transform(y_test)
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))