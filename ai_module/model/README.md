# Heart Disease Prediction Model Training and Evaluation

This module is designed to train and evaluate a machine learning model for predicting heart disease. It leverages patient data for model training and evaluation, and supports two types of models: **RandomForest** and **Logistic Regression**. The module includes data preprocessing, hyperparameter tuning, probability calibration, and performance evaluation.

## Overview

The module performs the following steps:
- **Data Preprocessing**:  
  - Standardizes feature values using a `StandardScaler`.
  - Converts categorical labels to numerical format with `LabelEncoder`.
  - Balances the training dataset using SMOTE to address class imbalance.
  
- **Model Training**:  
  - For **Logistic Regression**, the model is trained directly with parallel computation enabled.
  - For **RandomForest**, hyperparameter tuning is performed using `GridSearchCV` to find the best configuration (number of trees, maximum depth, and minimum samples split). After tuning, the model is calibrated using `CalibratedClassifierCV` to improve the reliability of the predicted probabilities.
  
- **Evaluation**:  
  - The model is evaluated on a validation dataset with metrics such as accuracy, classification report, and confusion matrix.
  - The trained model is saved as a `.pkl` file in the specified output directory.
  - Finally, the module loads a test dataset, applies the same preprocessing, and evaluates the model performance on this unseen data.

## Requirements

- Python 3.6+
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [joblib](https://joblib.readthedocs.io/)

Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn imbalanced-learn joblib
```

## File Structure

- **Module File (e.g., `model_training.py`)**:  
  Contains the main functions for training and evaluating the model, including:
  - `train_and_evaluate_model`: Trains the selected model and prints evaluation metrics.
  - The `if __name__ == "__main__":` block loads the training, validation, and test datasets, runs training, evaluates the model, and prints the results.

- **Datasets**:  
  The module expects preprocessed datasets saved as `.npy` files:
  - `X_train.npy`, `y_train.npy` – Training data.
  - `X_val.npy`, `y_val.npy` – Validation data.
  - `X_test.npy`, `y_test.npy` – Test data.
  
  These files should be located in the `../dataset/processed_dataset` directory (or update the paths accordingly).

## How to Run

1. **Prepare the Data**:  
   Ensure that the training, validation, and test datasets are saved as `.npy` files in the expected directory (e.g., `../dataset/processed_dataset`).

2. **Run the Module**:  
   Execute the module from the command line:
   ```bash
   python model_training.py
   ```
   This will train the model (using either RandomForest or Logistic Regression based on the `model_type` parameter), evaluate it on the validation set, and then evaluate the performance on the test set. The trained model is saved in the `model` directory.

3. **Model Output**:  
   The module will output:
   - Validation and test accuracy.
   - Classification reports and confusion matrices.
   - A saved model file (e.g., `heart_disease_model_random_forest.pkl`).

## Functions Description

### `train_and_evaluate_model`
- **Parameters**:
  - `X_train`, `X_val`: Feature matrices for training and validation.
  - `y_train`, `y_val`: Corresponding labels.
  - `output_dir` (str): Directory to save the trained model.
  - `model_type` (str): Type of model to train (`'random_forest'` or `'logistic_regression'`).
- **Returns**:
  - `model`: The trained machine learning model.
  - `le`: The `LabelEncoder` used to encode class labels.
  - `scaler`: The `StandardScaler` used for feature normalization.
- **Process**:
  - Encodes labels, standardizes data, handles class imbalance with SMOTE.
  - For RandomForest, performs hyperparameter tuning with `GridSearchCV` and calibrates probabilities.
  - Evaluates model performance on the validation set and prints the results.
  - Saves the trained model as a `.pkl` file.

## Hyperparameter Tuning and Calibration

For the RandomForest model, a grid search is used to tune the following parameters:
- **n_estimators**: Number of trees in the forest (e.g., 100, 300).
- **max_depth**: Maximum depth of the trees (e.g., 10, 15).
- **min_samples_split**: Minimum samples required to split an internal node (e.g., 1, 3).

After selecting the best model from the grid search, the model is wrapped with `CalibratedClassifierCV` to adjust the probability estimates using the sigmoid method.

## Additional Notes

- Ensure that the file paths and directory names in the code match your local environment.
- The module is designed to utilize all available CPU cores (`n_jobs=-1`) for faster computation.
- You can switch between model types by setting the `model_type` parameter in the main block.
