# Passenger No-Show Prediction using PySpark

## 📌 Project Overview
This project predicts whether a passenger will cancel their booking using machine learning techniques implemented in PySpark.

## 🚀 Features
- Data preprocessing and cleaning
- Feature engineering using StringIndexer
- Logistic Regression model
- Evaluation using Accuracy and ROC-AUC
- Visualization:
  - Confusion Matrix
  - ROC Curve
  - Cancellation Distribution

## 📊 Dataset
The dataset contains:
- From, To
- VehicleType, VehicleClass
- TripReason
- Price, CouponDiscount
- Domestic
- Target: Cancel (0/1)

## ⚙️ Technologies Used
- Python
- PySpark
- Scikit-learn
- Matplotlib

## 📈 Model
Logistic Regression is used for binary classification.

## 📌 Results
- Good prediction accuracy
- Effective classification performance
- Clear visualization insights

## ▶️ How to Run

### 1. Install dependencies
pip install pyspark matplotlib scikit-learn

### 2. Open Command Prompt (CMD)

### 3. Navigate to project folder
cd path\to\your\project

### 4. Run using PySpark
pyspark

### 5.Run directly using spark-submit
spark-submit main.py
