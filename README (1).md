# 🩺 Diabetes Classification Using Logistic Regression

This project focuses on building a *Diabetes Classification Model* using *Logistic Regression*. The notebook covers exploratory data analysis, data preprocessing, feature analysis, and model evaluation.

## 📌 Project Overview

The goal is to predict whether a patient has diabetes based on medical and demographic features.

The project focuses not only on training the model, but also on understanding and preparing the dataset before applying Logistic Regression.

## 🔍 Data Analysis & Preprocessing

The notebook includes the following steps:

### 1. Missing Values Analysis

* Checked the dataset for missing (NaN) values.
* Identified columns containing missing or invalid values.
* Applied appropriate preprocessing techniques where needed.

### 2. Distribution Analysis

* Analyzed the distribution of numerical features.
* Applied *Log Transformation* to highly skewed features where appropriate.
* Compared feature distributions before and after transformation.

Log transformation can help reduce right skewness and make feature distributions more suitable for machine learning models.

### 3. Class Imbalance Analysis

* Checked the distribution of the target variable.
* Identified whether the dataset was imbalanced.
* Used *class weights* in Logistic Regression to give more importance to the minority class.

python
LogisticRegression(class_weight="balanced")


Using balanced class weights helps the model pay greater attention to minority-class samples without simply duplicating observations.

### 4. Multicollinearity Analysis

* Analyzed relationships between numerical features.
* Used *correlation analysis* to identify highly correlated variables.
* Evaluated potential multicollinearity between predictors.

Multicollinearity can make Logistic Regression coefficients unstable and harder to interpret.

### 5. Feature Scaling

Since Logistic Regression is sensitive to feature magnitude, numerical features were scaled before training.

python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


### 6. Logistic Regression

The processed dataset was used to train a Logistic Regression classification model.

python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)


## 📊 Model Evaluation

The model can be evaluated using several classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Special attention is given to *Recall*, since correctly identifying diabetic patients is an important consideration in this classification problem.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## 📁 Project Structure

text
diabetes-classification-logistic-regression/
│
├── diabetes_classification.ipynb
├── README.md
└── dataset/
    └── diabetes.csv


## 🎯 Key Concepts

This project demonstrates practical applications of:

* Exploratory Data Analysis (EDA)
* Missing Value Analysis
* Distribution Analysis
* Log Transformation
* Class Imbalance Handling
* Class Weighting
* Correlation Analysis
* Multicollinearity
* Feature Scaling
* Logistic Regression
* Classification Model Evaluation

## 🚀 Conclusion

This project demonstrates an end-to-end machine learning workflow for diabetes classification, starting from data exploration and preprocessing and progressing to model training and 
