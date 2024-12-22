import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preprocess_data(file_path, output_dir="processed_dataset", outlier_threshold=1.5):
    """
    Loads, preprocesses, and prepares data from a CSV file, saving results to the specified directory.

    Args:
        file_path (str): Path to the CSV file.
        output_dir (str): Name of the output directory for processed data.
        outlier_threshold (float): Coefficient for outlier identification (IQR).

    Returns:
        tuple: A tuple containing:
            - X (np.array): Features before splitting.
            - y (np.array): Labels before splitting.
        or None if there is a file loading error.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Handle missing data using imputation
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data.loc[:, col] = data[col].fillna(data[col].median())
            else:
                data.loc[:, col] = data[col].fillna(data[col].mode()[0])

    # Identify and remove outliers based on BMI using the IQR method BEFORE encoding target variable
    Q1 = data['BMI'].quantile(0.25)
    Q3 = data['BMI'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data['BMI'] < (Q1 - outlier_threshold * IQR)) | (data['BMI'] > (Q3 + outlier_threshold * IQR))]
    data_filtered = data[~((data['BMI'] < (Q1 - outlier_threshold * IQR)) | (data['BMI'] > (Q3 + outlier_threshold * IQR)))]
    outliers.to_csv(os.path.join(output_dir, "outliers_bmi.csv"), index=False)  # Save outliers to CSV

    # Encode the target variable using LabelEncoder
    le_heart_disease = LabelEncoder()
    data_filtered.loc[:, 'HeartDisease'] = le_heart_disease.fit_transform(data_filtered['HeartDisease'])

    # Convert categorical variables to numerical using Label Encoding
    categorical_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    for col in categorical_cols:
        le = LabelEncoder()
        data_filtered.loc[:, col] = le.fit_transform(data_filtered[col])  # Save encoder

    # Prepare data for machine learning
    X = data_filtered.drop('HeartDisease', axis=1).values  # Features
    y = data_filtered['HeartDisease'].values  # Labels
    return X, y


def test_data_split(X, y, test_size=0.3, random_state=42):
    """
    Tests the correctness of data splitting into training, validation, and test sets.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Label vector.
        test_size (float): Proportion of the test data.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - tests_passed (bool): True if all tests passed successfully, False otherwise.
            - X_train (np.array): Training data (features).
            - X_val (np.array): Validation data (features).
            - X_test (np.array): Test data (features).
            - y_train (np.array): Training labels.
            - y_val (np.array): Validation labels.
            - y_test (np.array): Test labels.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    n_total = len(X)
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    tests_passed = True

    if X_test.shape[0] != y_test.shape[0]:
        print("Error: The number of rows in X_test and y_test is different.")
        tests_passed = False

    if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray):
        print("Error: X_test or y_test are not NumPy arrays.")
        tests_passed = False

    if not np.all(np.isin(y_test, [0, 1])):  # Check if y_test contains only 0 and 1
        print("Error: y_test contains values outside the range [0, 1].")
        tests_passed = False

    if n_total != n_train + n_val + n_test:
        print(f"Error: The sum of the number of samples after splitting ({n_train} + {n_val} + {n_test} = {n_train + n_val + n_test}) does not match the number of samples before splitting ({n_total}).")
        tests_passed = False

    if tests_passed:
        print("All data split tests passed successfully.")
        print(f"Set sizes: Training: {n_train}, Validation: {n_val}, Test: {n_test}, Total: {n_total}")

    return tests_passed, X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    file_path = "heart_2020_cleaned.csv"  # Replace with your file path
    output_directory = "processed_dataset"
    X, y = preprocess_data(file_path, output_dir=output_directory)

    if X is not None and y is not None:
        tests_passed, X_train, X_val, X_test, y_train, y_val, y_test = test_data_split(X, y) # Execute tests and get data splits
        if tests_passed:
            print("Data has been processed and tests passed successfully.")
            # Scale numerical data using StandardScaler and save the scaler AFTER splitting
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            joblib.dump(scaler, os.path.join(output_directory, 'scaler.pkl'))  # Save scaler

            # Save data to .npy files AFTER scaling
            np.save(os.path.join(output_directory, 'X_train.npy'), X_train)
            np.save(os.path.join(output_directory, 'X_val.npy'), X_val)
            np.save(os.path.join(output_directory, 'X_test.npy'), X_test)
            np.save(os.path.join(output_directory, 'y_train.npy'), y_train)
            np.save(os.path.join(output_directory, 'y_val.npy'), y_val)
            np.save(os.path.join(output_directory, 'y_test.npy'), y_test)
            joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test), os.path.join(output_directory, 'data.joblib')) # Zapis całych danych do jednego pliku
        else:
            print("Data split tests detected errors.")
    else:
        print("Data processing failed.")
