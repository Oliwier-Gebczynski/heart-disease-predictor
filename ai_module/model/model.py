import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def train_and_evaluate_model(X_train, X_val, y_train, y_val, output_dir="model", model_type='logistic_regression'):
    """
    Trenuje model i ocenia go na zbiorze walidacyjnym.
    Zapisuje model do pliku .pkl w folderze /model.
    """
    # Przekonwertuj etykiety na liczby całkowite
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    # Balansowanie klas
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Inicjalizacja modelu
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        param_grid = {
            'C': [0.01, 1, 100],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    else:
        raise ValueError("Invalid model_type. Choose from 'logistic_regression', 'random_forest'.")

    # Dostrojenie hiperparametrów z wykorzystaniem wszystkich rdzeni
    with tqdm(total=10, desc="Randomized Search Progress") as pbar:
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=8,
            cv=3,
            scoring='accuracy',
            verbose=1,
            random_state=42,
            n_jobs=-1  # Kluczowa zmiana!
        )
        random_search.fit(X_train, y_train)
        pbar.update(10)

    # Najlepszy model
    best_model = random_search.best_estimator_

    # Predykcja i ocena
    y_val_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # Zapisanie modelu
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'heart_disease_model_{model_type}.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    return best_model

if __name__ == "__main__":
    output_directory = "../dataset/processed_dataset"
    model_directory = "model"
    
    # Wczytanie danych
    X_train = np.load(os.path.join(output_directory, 'X_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(output_directory, 'X_val.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(output_directory, 'y_train.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(output_directory, 'y_val.npy'), allow_pickle=True)

    print("Unique values in y_train:", np.unique(y_train))
    print("Unique values in y_val:", np.unique(y_val))

    # Trenowanie modelu
    model = train_and_evaluate_model(X_train, X_val, y_train, y_val, output_dir=model_directory, model_type='random_forest')

    # Ocena na zbiorze testowym
    X_test = np.load(os.path.join(output_directory, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(output_directory, 'y_test.npy'), allow_pickle=True)
    y_test = LabelEncoder().fit_transform(y_test)
    
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))