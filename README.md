# California House Price Prediction
An end-to-end machine learning application for predicting California house prices.
The project demonstrates data preprocessing, model training, evaluation, and deployment through an interactive Streamlit interface.

---

# Overview
This project uses the California Housing dataset to predict the median house value based on geographical, demographic, and housing-related features.
Rather than focusing on model novelty, the emphasis is on:

- Clean data preparation

- Reproducible model training

- Objective model evaluation

- Practical deployment for inference

---

# Key Features

- CSV upload with fallback to default dataset

- Automated data cleaning and encoding

- Model comparison using standard regression metrics

- Feature importance analysis

---

# Models
Two models are implemented:

Linear Regression : 
Used as a baseline to establish reference performance.

Random Forest Regressor : 
Selected for deployment due to improved performance on non-linear relationships.

---

# Evaluation Metrics

Model performance is evaluated using:

- Mean Squared Error (MSE) - penalizes large errors

- Mean Absolute Error (MAE) - real-world deviation

- R² Score - variance

---

# Web Application (Streamlit)

The Streamlit app is divided into multiple tabs:

- Upload Dataset – Upload CSV or use default dataset

- Preprocessed Dataset – View cleaned data

- Summary Statistics – Dataset overview & stats

- Training ML Models – Train and evaluate models

- Evaluation – Feature importance visualization

- Prediction – Predict house prices using user input

---

# Tech Stack

- Python

- Pandas

- NumPy

- Scikit-learn

- Streamlit

- Pickle
