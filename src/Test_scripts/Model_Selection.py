''' TRAIN AND TEST SELECTED MODEL WITH HYPER PARAMETER TUNING '''

import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pickle


def model_selection():

    DATA_PATH = '/Users/sarveshdhond/Projects/House_price_predictor/dataset/'
    MODEL_PATH = '/Users/sarveshdhond/Projects/House_price_predictor/model/XGB_MODEL_TEST.pkl'

    train_df = pd.read_csv(DATA_PATH+'featured_data/test_featured_train_data.csv')
    test_df = pd.read_csv(DATA_PATH+'featured_data/test_featured_test_data.csv')
    
    X_train = train_df.drop(columns=['price'])
    y_train = train_df['price']
    X_test = test_df.drop(columns=['price'])
    y_test = test_df['price']
    
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
    study.optimize(objective , n_trials=5)
    best_params = study.best_params
    print(study.best_params)
    print('Model hyper tuning completed')
    print('')

    print('Starting Model retest')
    MODEL = XGBRegressor(**best_params)
    MODEL.fit(X_train, y_train)
    y_pred = MODEL.predict(X_test)
    final_test_mae = mean_absolute_error(y_test, y_pred)

    print(f'MODEL MAE : {final_test_mae}')
    print('Model retest completed')
    print('')

    with open (MODEL_PATH, 'wb') as f:
        pickle.dump(MODEL, f)
    
    print('Model Selection completed')
    print('')
    print('Hyper tuned model saved')
    print('')

    return final_test_mae

if __name__ == '__main__':
    model_selection()

