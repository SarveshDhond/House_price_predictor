''' DATA ANALYSIS AND CLEANING '''

import pandas as pd


def data_analysis():

    DATA_PATH = '/Users/sarveshdhond/Projects/House_price_predictor/dataset/'
    df = pd.read_csv(DATA_PATH+'raw_data/house_raw.csv')
    city_df = pd.read_csv(DATA_PATH+'raw_data/city_raw.csv')

    def clean_text(x):
        x = x.lower()
        x = x.strip()
        x = x.split('-')[0]
        x = x.split('_')[0]
        x = x.replace('.','')
        x = x.replace('city','')
        return x

    df['city_full'] = df['city_full'].apply(clean_text)
    city_df['city_full'] = city_df['City'].apply(clean_text)
    city_df.drop(columns=['City'], inplace=True, errors='ignore')

    df = pd.merge(df , city_df , on='city_full' , how='left')

    df = df.dropna()
    df = df.drop_duplicates(subset=df.drop(columns=['date']))

    df.columns=['date', 'median_sale_price', 'median_list_price', 'median_ppsf',
        'median_list_ppsf', 'homes_sold', 'pending_sales', 'new_listings',
        'inventory', 'median_dom', 'avg_sale_to_list', 'sold_above_list',
        'off_market_in_two_weeks', 'city', 'zipcode', 'year', 'bank', 'bus',
        'hospital', 'mall', 'park', 'restaurant', 'school', 'station',
        'supermarket', 'Total_Population', 'Median_Age', 'Per_Capita_Income',
        'Total_Families_Below_Poverty', 'Total_Housing_Units', 'Median_Rent',
        'Median_Home_Value', 'Total_Labor_Force', 'Unemployed_Population',
        'Total_School_Age_Population', 'Total_School_Enrollment',
        'Median_Commute_Time', 'price', 'city_full', 'Latitude', 'Longitude']

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df.drop(columns=['date'], inplace=True, errors='ignore')

    train_df = df[df['year'] < 2020]
    train_df = train_df.copy()
    test_df = df[df['year'] >= 2020]
    test_df = test_df.copy()


    print('Data analysis completed')
    print('')
    print('Train and Test dataset saved')
    print('')

    return train_df, test_df


