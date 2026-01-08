''' TRAIN AND TEST SELECTED MODEL WITH HYPER PARAMETER TUNING '''

import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pickle


def model_selection(featured_train_df, featured_test_df):

    MODEL_PATH = '/Users/sarveshdhond/Projects/House_price_predictor/model/XGB_MODEL_FINAL.pkl'
    
    X_train = featured_train_df.drop(columns=['price'])
    y_train = featured_train_df['price']
    X_test = featured_test_df.drop(columns=['price'])
    y_test = featured_test_df['price']
    
    def objective(trial):
       n_estimators = trial.suggest_int('n_estimators', 100 , 1000)
       max_depth = trial.suggest_int('max_depth', 1, 10)
       learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3)
       eta = trial.suggest_float('eta', 0.1 , 0.5)
       subsample = trial.suggest_int('subsample', 0 , 1)
       random_state = 42
       
       model = XGBRegressor(
          n_estimators=n_estimators,
          max_depth=max_depth,
          learning_rate=learning_rate,
          eta=eta,
          subsample=subsample,
          random_state=random_state
          )
       
       model.fit(X_train , y_train, verbose=False)
       y_pred = model.predict(X_test)
       mae = mean_absolute_error(y_test , y_pred)
       return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective , n_trials=100)
    best_params = study.best_params
    print(study.best_params)
    print('Model hyper tuning completed')
    print('')

    print('Starting Model retest')
    MODEL = XGBRegressor(**best_params)
    MODEL.fit(X_train, y_train)
    y_pred = MODEL.predict(X_test)
    final_mae = mean_absolute_error(y_test, y_pred)

    print(f'MODEL MAE : {final_mae}')
    print('Model retest completed')
    print('')

    with open (MODEL_PATH, 'wb') as f:
        pickle.dump(MODEL, f)
    
    print('Model Selection completed')
    print('')
    print('Hyper tuned model saved')
    print('')

    return final_mae
