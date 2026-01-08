''' FEATURE ENGINEERING '''

import pandas as pd
from category_encoders import TargetEncoder

def feature_engineering():
    DATA_PATH = '/Users/sarveshdhond/Projects/House_price_predictor/dataset/'
    train_df = pd.read_csv(DATA_PATH+'processed_data/test_processed_train_data.csv')
    test_df = pd.read_csv(DATA_PATH+'processed_data/test_processed_test_data.csv')

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = train_df[train_df['median_list_price'] < 7000000]
    test_df = test_df[test_df['median_list_price'] < 7000000]

    freq = train_df['zipcode'].value_counts()
    train_df['zipcode_freq'] = train_df['zipcode'].map(freq)
    test_df['zipcode_freq'] = test_df['zipcode'].map(freq).fillna(freq.min())

    TE = TargetEncoder(cols=['city_full'], smoothing=10)
    train_df['cityTE'] = TE.fit_transform(train_df['city_full'], train_df['price'])
    test_df['cityTE'] = TE.transform(test_df['city_full'])

    columns_to_drop = ['zipcode','city_full','city']
    train_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    high_vif = ['Total_School_Enrollment','Total_School_Age_Population','Total_Population','Total_Labor_Force','Median_Commute_Time','Total_Families_Below_Poverty','Total_Housing_Units','new_listings']
    train_df.drop(columns=high_vif, inplace=True, errors='ignore')
    test_df.drop(columns=high_vif, inplace=True, errors='ignore')

    train_df.to_csv(DATA_PATH+'featured_data/test_featured_train_data.csv', index=False)
    test_df.to_csv(DATA_PATH+'featured_data/test_featured_test_data.csv', index=False)

    featured_train_df = train_df.copy()
    featured_test_df = test_df.copy()

    print('Feature Engineering completed')
    print('')
    print('Feature trained and feature tested dataset saved')
    print('')

    return featured_train_df, featured_test_df

if __name__ == '__main__':
    feature_engineering()

