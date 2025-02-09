# Heart Disease Prediction Project

This project is dedicated to developing a machine learning pipeline for predicting heart disease based on patient data. It encompasses multiple modules including data preprocessing, model training and evaluation, and prediction. The goal is to deliver a robust solution that not only builds accurate predictive models but also provides well-calibrated probability estimates to support clinical decision-making.

## Overview

The project consists of the following main components:

- **Data Preprocessing Module**  
  This module handles the ingestion and cleaning of raw patient data from CSV files. It performs:
  - Missing value imputation using the median (for numeric features) or mode (for categorical features).
  - Label encoding of categorical features and the target variable.
  - Optional outlier detection and removal.
  - Splitting and saving of the processed data as NumPy arrays for further use.

- **Model Training and Evaluation Module**  
  This module is responsible for training machine learning models using the preprocessed data. It supports two model types:
  - **RandomForest Classifier** (with hyperparameter tuning via GridSearchCV and probability calibration using CalibratedClassifierCV).
  - **Logistic Regression** (using parallel computation for speed).
  
  The module evaluates the model using standard metrics (accuracy, classification report, and confusion matrix) on a validation set, and the best model is saved to disk.

- **Prediction Module**  
  This module loads the saved model and processes new input data (using the same preprocessing pipeline) to predict the probability of heart disease for each patient. The results are printed to the terminal and saved as a CSV file.

## File Structure

```
heart-disease-prediction/
├── data_preprocessing.py      # Script for data loading, cleaning, and preprocessing
├── model_training.py          # Script for training and evaluating machine learning models
├── prediction.py              # Script for making predictions with the trained model
├── patients_data.csv          # Input CSV file containing raw patient data
├── processed_dataset/         # Directory for storing preprocessed data (.npy files)
├── model/                     # Directory for saving the trained model (.pkl file)
├── result/                    # Directory for saving prediction results (CSV file)
└── README.md                  # This file
```

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

## How to Run the Project

1. **Data Preprocessing**  
   Run the data preprocessing script to clean and prepare the raw CSV data:
   ```bash
   python data_preprocessing.py
   ```
   This script reads `patients_data.csv`, handles missing values, encodes categorical variables, and saves the processed features (`X.npy`) and labels (`y.npy`) to the `processed_dataset` directory.

2. **Model Training and Evaluation**  
   Train and evaluate the predictive model (choose between RandomForest and Logistic Regression by setting the `model_type` parameter) by running:
   ```bash
   python model_training.py
   ```
   The script loads the preprocessed data, applies scaling and SMOTE to balance classes, and then trains the model. It outputs evaluation metrics and saves the trained model (e.g., `heart_disease_model_random_forest.pkl`) to the `model` directory.

3. **Making Predictions**  
   Use the prediction script to run new data through the trained model:
   ```bash
   python prediction.py
   ```
   This script loads the saved model, preprocesses the input data (if needed), predicts the probability---

This README provides a comprehensive overview of the project, detailing its modules, usage, and file structure. If you have any questions or require further assistance, please refer to the inline comments within the source code or contact the project maintainer. of heart disease for each patient, prints the results to the terminal, and saves them as `predictions.csv` in the `result` directory.

## Example Workflow

After following the steps above, your workflow might look like this:

- **Preprocessing**:  
  Process raw data and generate `X.npy` and `y.npy` files.
  
- **Training**:  
  The model is trained and validated on the preprocessed data. Evaluation metrics (accuracy, classification report, and confusion matrix) are printed, and the best model is saved.

- **Prediction**:  
  The model predicts heart disease probabilities on new data. Predictions for each patient are output to the terminal and saved as a CSV file.

## Additional Notes

- **Customization**:  
  You can modify parameters (e.g., model type, hyperparameters, outlier thresholds) by editing the corresponding sections in the scripts.
  
- **Parallel Computation**:  
  The project utilizes all available CPU cores (`n_jobs=-1`) in scikit-learn functions to speed up computations.

- **Extensibility**:  
  The modular structure of the project makes it easy to replace or extend individual components, such as experimenting with different preprocessing techniques or trying other machine learning algorithms.

## Contributing

Contributions to improve this project are welcome. Please feel free to submit issues or pull requests if you find any bugs or have suggestions for enhancements.

## License

This project is open source and available under the MIT License.