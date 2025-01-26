import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def preprocess_data(file_path, output_dir="processed_dataset"):
    """
    Loads and prepares data from a CSV file, saving results to the specified directory.

    Args:
        file_path (str): Path to the CSV file.
        output_dir (str): Name of the output directory for processed data.

    Returns:
        tuple: A tuple containing:
            - X (np.array): Features.
            - y (np.array): Labels.
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

    # Encode the target variable using LabelEncoder
    le_heart_disease = LabelEncoder()
    data.loc[:, 'HeartDisease'] = le_heart_disease.fit_transform(data['HeartDisease'])

    # Convert categorical variables to numerical using Label Encoding
    categorical_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    for col in categorical_cols:
        le = LabelEncoder()
        data.loc[:, col] = le.fit_transform(data[col])

    # Prepare data for machine learning
    X = data.drop('HeartDisease', axis=1).values  # Features
    y = data['HeartDisease'].values  # Labels

    return X, y

def predict_with_model(model_path, input_data):
    """
    Predicts the probability of heart disease using the trained model.

    Args:
        model_path (str): Path to the trained model (.pkl file).
        input_data (np.array): Input data for prediction.

    Returns:
        np.array: Predicted probabilities of heart disease.
    """
    # Load the model
    model = joblib.load(model_path)

    # Predict probabilities
    predictions = model.predict_proba(input_data)[:, 1]  # Probability of class 1 (disease)
    return predictions

if __name__ == "__main__":
    # File paths
    file_path = "patients_data.csv"  # Path to the input CSV file
    model_path = "./model/model/heart_disease_model_random_forest.pkl"  # Path to the trained model
    output_directory = "./result"

    # Preprocess the data
    X, y = preprocess_data(file_path, output_dir=output_directory)

    if X is not None and y is not None:
        # Save the entire dataset to a single .npy file
        np.save(os.path.join(output_directory, 'X.npy'), X)
        np.save(os.path.join(output_directory, 'y.npy'), y)
        print("Data saved to X.npy and y.npy in the processed_dataset directory.")

        # Predict probabilities using the trained model
        predictions = predict_with_model(model_path, X)

        # Print predictions for each patient
        for i, prediction in enumerate(predictions):
            print(f"Patient {i + 1}: Probability of heart disease: {prediction:.4f}")

        # Save predictions to a CSV file
        output_csv = os.path.join(output_directory, 'predictions.csv')
        pd.DataFrame({
            'PatientID': range(1, len(predictions) + 1),
            'HeartDiseaseProbability': predictions
        }).to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}.")
    else:
        print("Data processing failed.")