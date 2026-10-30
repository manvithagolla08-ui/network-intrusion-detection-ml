# Network Intrusion Detection – ML Challenge

## 1. Project Overview

This project focuses on detecting and classifying network traffic into four categories:

- Benign
- Availability
- Reconnaissance
- Spoofing

The objective is to build a machine learning model that identifies different types of network traffic and detects potential intrusions.

## 2. Dataset

The project uses the provided training and testing datasets.

- Training data: `train.csv`
- Testing data: `test.csv`

The target column is `Target`. The `id` column is used to identify each test record.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Loaded the datasets using Pandas.
- Separated input features and target labels.
- Removed the `id` and `Target` columns from the training features.
- Handled categorical features such as `proto` and `service`.
- Used the same categorical encoding across training and testing data.

## 4. Machine Learning Algorithm

LightGBM (Light Gradient Boosting Machine) was used for multiclass classification.

Two LightGBM models were combined using a weighted probability blend:

- Model 1: Original LightGBM model.
- Model 2: Regularized LightGBM model.
- Blending weights: 80% Model 1 and 20% Model 2.

The final class is selected based on the highest blended prediction probability.

## 5. Evaluation Metric

The competition uses Macro F1-score.

Macro F1 calculates the F1-score separately for each class and takes their average, giving equal importance to all classes.

## 6. Prediction and Submission

The trained models generate predictions for the test dataset.

The final predictions are stored in `submission.csv` with exactly two columns:

- `id`
- `Target`

## 7. Execution

Install the required libraries:

```bash
pip install pandas numpy scikit-learn lightgbm
```

Place `train.csv` and `test.csv` in the project folder and run:

```bash
python main.py
```

The script generates `generated_submission.csv`.

## 8. Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM

## 9. Note

The submitted prediction file is the best-performing candidate obtained during experimentation. The reported public leaderboard score was 0.84452.