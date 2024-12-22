# Heart Disease Prediction - Data Preprocessing Module

This repository contains a Python script (`data_preprocessing.py`) designed to preprocess the [heart disease dataset](`https://www.kaggle.com/code/mushfirat/heartdisease-eda-prediction`) for machine learning tasks. The script handles missing values, removes outliers (based on BMI), encodes categorical variables, and splits the data into training, validation, and test sets.

## Overview

The `data_preprocessing.py` script performs the following steps:

1.  **Data Loading:** Loads the data from a CSV file (`heart_2020_cleaned.csv`).
2.  **Missing Value Imputation:** Fills missing values using the median for numerical columns and the mode for categorical columns.
3.  **Outlier Removal:** Identifies and removes outliers based on the BMI column using the Interquartile Range (IQR) method. Outliers are saved to `outliers_bmi.csv`.
4.  **Categorical Encoding:** Converts categorical features into numerical representations using Label Encoding.
5.  **Target Variable Encoding:** Encodes the target variable (`HeartDisease`) using Label Encoding.
6.  **Data Splitting:** Splits the data into training (70%), validation (15%), and test (15%) sets.
7.  **Data Scaling:** Scales numerical features using `StandardScaler` *after* splitting the data.
8.  **Data Saving:** Saves the processed data (scaled features and encoded labels) as NumPy `.npy` files and all splits to a single `.joblib` file. Encoders and the scaler are also saved.

## Files

*   `data_preprocessing.py`: The Python script for data preprocessing.
*   `heart_2020_cleaned.csv`: The input dataset (should be placed in the same directory or provide the correct path).
*   `processed_dataset/`: The output directory containing:
    *   `X_train.npy`, `X_val.npy`, `X_test.npy`: Scaled feature matrices for training, validation, and testing.
    *   `y_train.npy`, `y_val.npy`, `y_test.npy`: Encoded labels for training, validation, and testing.
    *   `data.joblib`: All data splits in one file
    *   `label_encoder_*.pkl`: Saved LabelEncoders for each categorical column.
    *   `label_encoder_HeartDisease.pkl`: Saved LabelEncoder for the target variable.
    *   `scaler.pkl`: Saved StandardScaler.
    *   `outliers_bmi.csv`: The file containing removed outliers based on BMI.

## Usage

1.  Place the `heart_2020_cleaned.csv` file in the same directory as `data_preprocessing.py` or provide the correct file path as an argument.
2.  Run the script:

    ```bash
    python3 data_preprocessing.py
    ```

    You can specify a different output directory using the `--output_dir` argument and the outlier threshold with `--outlier_threshold`:
    ```bash
    python3 data_preprocessing.py --output_dir my_processed_data --outlier_threshold 1.0
    ```
3. The processed data will be saved in the `processed_dataset` directory (or your specified directory).

## Requirements

*   Python 3
*   pandas
*   NumPy
*   scikit-learn
*   joblib

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn joblib
```

## Example
After running the script, you can load the processed data in your machine learning scripts like this:

```python
import numpy as np
import joblib
import os

output_directory = "processed_dataset" # or your output directory
X_train = np.load(os.path.join(output_directory, 'X_train.npy'))
y_train = np.load(os.path.join(output_directory, 'y_train.npy'))

# or load all data at once
(X_train, X_val, X_test, y_train, y_val, y_test) = joblib.load(os.path.join(output_directory, 'data.joblib'))

scaler = joblib.load(os.path.join(output_directory, 'scaler.pkl'))
# ... use the data for training your model
```


