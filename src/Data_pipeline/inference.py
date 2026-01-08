from src.Data_pipeline.Data_Analysis import data_analysis
from src.Data_pipeline.Feature_Engineering import feature_engineering
from src.Data_pipeline.Model_Selection import model_selection

import pandas as pd
from category_encoders import TargetEncoder
import optuna
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pickle


train_df , test_df = data_analysis()
featured_train_df, featured_test_df = feature_engineering(train_df , test_df)
mae = model_selection(featured_train_df, featured_test_df)

print(mae)
