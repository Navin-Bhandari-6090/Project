# Cancer Prediction Using Machine Learning

This project focuses on predicting cancer diagnosis (Benign/Malignant) using various machine learning models. The system includes end-to-end data preprocessing, feature engineering, visualization, model training, and evaluation to generate reliable predictions.

## 📌 Project Overview

This project uses a structured ML pipeline to analyze medical data and classify the diagnosis based on numerical features. It demonstrates concepts like data cleaning, imputation, scaling, model training, evaluation, and visualization.

## 📂 Dataset Description

* The dataset is loaded from **data.csv**
* Contains patient-related diagnostic measurements
* The target variable is **diagnosis**
* The dataset includes both numerical and categorical columns
* Unnecessary ID column was removed for cleaner processing


## 🔧 Technologies & Libraries Used

* Python
* NumPy,Pandas
* Matplotlib,Seaborn
* scikit-learn (sklearn)

  * SimpleImputer
  * ColumnTransformer
  * Pipeline
  * Train/Test Split
  * StandardScaler
  * KNN Classifier
  * SVM Classifier



## 🧹 Data Preprocessing Steps

* Removed irrelevant features (e.g., `id`)
* Identified numerical and categorical columns
* Checked for missing values
* Applied imputations using **ColumnTransformer**:

  * Mean imputation for numeric columns
  * Mode imputation for categorical column
* Updated dataset with transformed values
* Performed train/test split (80% training, 20% testing)


## 📊 Data Visualization

* Histogram plots for all numeric features
* KDE curves added to identify distribution shape
* Plots help confirm Gaussian distribution, supporting the use of "StandardScaler"

## ⚙️ Feature Scaling

* Used "StandardScaler" to normalize feature values
* Scaling ensures better performance for distance-based algorithms like KNN and SVM

## 🤖 Machine Learning Models Used

### 1. K-Nearest Neighbors (KNN)

* Used `n_neighbors = 1`
* Trained on scaled features
* Generated accuracy score and predictions

### 2. Support Vector Machine (SVM)

* Used RBF kernel
* Trained on scaled dataset
* Achieved high accuracy for classification


## 📈 Model Evaluation

* Accuracy score calculated for each model
* Predictions printed to compare performance
* KNN and SVM both tested on the same dataset

## 🎯 Results
* Successfully predicted cancer diagnosis using ML algorithms
* Models performed well after proper preprocessing and scaling

## 🚀 Future Improvements
* Add additional ML models (Random Forest, Logistic Regression, XGBoost)
* Deploy model using Flask / FastAPI
* Create a user-friendly web UI
* Perform hyperparameter tuning (GridSearch / RandomSearch)
* Use cross-validation for more stable performance



## How to Run the Project
* Ensure Python 3.x is installed
* Install required libraries
* Place **data.csv** in project folder
* Run the script to train and test the models


