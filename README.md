# 🏠 House_price_predictor
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## 📌 Project Overview
House price predictor is a machine learning project for predicting house prices based on various features. It includes data analysis, feature engineering, model comparison, hyperparameter tuning, and deployment as a FastAPI app. The final model uses XGBoost regression (after tuning) and serves predictions through a containerized API.

## 🔍 Data Analysis
- Data Import & Cleaning: Loaded the dataset, checked for missing values, and standardized entries (e.g. cleaned city names). Mapped each city to its latitude/longitude. Dropped duplicate rows and renamed columns for consistency.
- Feature Extraction: Split the date field into separate Year, Quarter, and Month features.
- Train/Test Split: Divided the data into training and testing sets for modeling.


## 🧪 Feature Engineering
- Outlier Detection: Applied Isolation Forest to identify anomalies. Isolation Forest isolates observations by random feature splits; anomalies tend to have shorter average path lengths in the tree ensemble. These outliers were removed from the training data.
- Categorical Encoding: Encoded the high-cardinality zip_code feature using frequency encoding. This replaces each category with its frequency, retaining statistical significance without exploding dimensionality.
- Multicollinearity: Computed Variance Inflation Factors (VIF) to detect highly correlated predictors. Features with high multicollinearity were dropped to improve model stability
- Final Datasets: Saved the cleaned and encoded training and testing feature sets for use in modeling.

## 🤖 Model Selection
We trained and evaluated several regressors. Key metrics (on the training data) were:


| Model | MAE |
| ---- | ---- |
| Dummy Regressor |	267098.01 |
| Linear Regression |	296,248.6 |
| Ridge Regression |	296,249.5 |
| Lasso Regression |	267,941.8 |
| ElasticNet |	258,898.3	|
| XGBoost Regressor |	41,548.6 |
| Tensorflow |	53,935.7 |

Among these, 🏆 XGBoost achieved by far the best performance (MAE ≈ 41,548; R² ≈ 0.939). This matches findings in the literature that XGBoost often outperforms traditional regressors on housing data. Based on these results, we selected XGBoost for further optimization.

## ⚙️ Hyperparameter Tuning
- We used Optuna for automated hyperparameter optimization of XGBoost
- Two scenarios were tested: training on the original target and on the log-transformed target. The model trained on the original prices yielded better results. The final tuned XGBoost model achieved a 🎯 Mean Absolute Error of ≈40,060 on the test set.


## 🚀 Deployment
- The trained model is deployed via a FastAPI app (app/main.py) that exposes a REST endpoint for price prediction.
- We containerized the app with Docker for easy deployment.
- The requirements.txt includes all dependencies.


## 📁 Project Structure
- app/ – FastAPI application code (main.py for the prediction endpoint).
- dataset/ – Raw, processed and feature engineered data files.
- model/ – The final saved XGBoost model after Optuna.
- notebook/ – Google colab notebooks covering EDA, preprocessing, and modeling steps.
- src/ – Reusable scripts for data cleaning, feature engineering and model training and testing.
- Dockerfile - Code to containerizing main, model and dependencies.
- requirements.txt – Contains all dependencies.











