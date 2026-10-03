# Customer Churn Prediction using ANN

An Artificial Neural Network (ANN) built with TensorFlow/Keras that predicts whether a bank customer is likely to leave (churn), deployed as an interactive Streamlit web app.

**Live demo:** [Open the Streamlit app](https://ann-classification-churn-aax8aqdqr6lnsdm3kbkwuy.streamlit.app/)

![App Screenshot](screenshot.png)

## Overview

Customer churn is when a customer stops using a company's service. Retaining existing customers is usually cheaper than acquiring new ones, so predicting churn helps a bank act early.

This project trains a feedforward neural network on the Churn Modelling dataset and serves predictions through a web interface. The user enters a customer's details and the app returns the churn probability along with a clear yes/no result.

## Features

- Binary classification of customer churn with a Keras ANN
- Preprocessing pipeline: label encoding, one-hot encoding and feature scaling
- Interactive Streamlit UI for entering customer details
- Displays the churn probability, not just the final label
- Trained model and preprocessors saved and loaded for inference

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow / Keras
- **Data processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib
- **Web app:** Streamlit

## Dataset

`Churn_Modelling.csv` contains records for 10,000 bank customers. Features used for prediction:

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated annual salary |

The target is `Exited` (1 = churned, 0 = stayed).

## Project Structure

```
ANN-Classification-Churn/
├── app.py                  # Streamlit web app
├── experiments.ipynb       # Data preprocessing and model training
├── predication.ipynb       # Testing predictions on sample input
├── Churn_Modelling.csv     # Dataset
├── model.h5                # Trained ANN model
├── Label_encoder.pkl       # Label encoder for Gender
├── one_hotencoder.pkl      # One-hot encoder for Geography
├── scaler.pkl              # StandardScaler for numeric features
├── requirements.txt        # Python dependencies
├── screenshot.png          # App screenshot used in this README
└── README.md
```

## How It Works

1. **Preprocessing:** Gender is label encoded, Geography is one-hot encoded, and all features are scaled with `StandardScaler`.
2. **Training:** The ANN is trained on the processed data in `experiments.ipynb`, and the model and encoders are saved.
3. **Inference:** `app.py` loads the saved model and preprocessors, applies the same transformations to the user's input, and predicts the churn probability.
4. **Output:** If the probability is above 0.5, the customer is flagged as likely to churn.

## Run Locally

1. Clone the repository:

```bash
   git clone https://github.com/TheR0HIT/ANN-Classification-Churn.git
   cd ANN-Classification-Churn
```

2. Create and activate a virtual environment (recommended):

```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows
```

3. Install the dependencies:

```bash
   pip install -r requirements.txt
```

4. Start the app:

```bash
   streamlit run app.py
```

5. Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Choose the customer's geography and gender.
2. Set age, tenure, number of products, and the credit card and active member options.
3. Enter balance, credit score and estimated salary.
4. Read the predicted churn probability and the result message.

## Future Improvements

- Hyperparameter tuning and a comparison against other models (Random Forest, XGBoost)
- Handling class imbalance (SMOTE or class weights)
- Model explainability with SHAP
- Evaluation metrics shown inside the app (precision, recall, ROC-AUC)

## Author

**Rohit Raj**
GitHub: [@TheR0HIT](https://github.com/TheR0HIT)
