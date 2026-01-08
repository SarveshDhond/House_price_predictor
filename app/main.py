# IMPORTS
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np


# LOAD MODEL
with open("/Users/sarveshdhond/Projects/House_price_predictor/model/XGB_MODEL_FINAL.pkl" , "rb") as f: 
    model = pickle.load(f)


# SET COLUMNS
EXPECTED_COLUMNS = ['median_sale_price', 'median_list_price', 'median_ppsf',
       'median_list_ppsf', 'homes_sold', 'pending_sales', 'inventory',
       'median_dom', 'avg_sale_to_list', 'sold_above_list',
       'off_market_in_two_weeks', 'year', 'bank', 'bus', 'hospital', 'mall',
       'park', 'restaurant', 'school', 'station', 'supermarket', 'Median_Age',
       'Per_Capita_Income', 'Median_Rent', 'Median_Home_Value',
       'Unemployed_Population', 'Latitude', 'Longitude', 'month',
       'quarter', 'zipcode_freq', 'cityTE']


# VALIDATE DATA
class ClientData(BaseModel): 
    median_sale_price: float
    median_list_price: float
    median_ppsf: float
    median_list_ppsf: float
    homes_sold: float
    pending_sales: float
    inventory: float
    median_dom: float
    avg_sale_to_list: float
    sold_above_list: float
    off_market_in_two_weeks: float
    year: int
    bank: float
    bus: float
    hospital: float
    mall: float
    park: float
    restaurant: float
    school: float
    station: float
    supermarket: float
    Median_Age: float
    Per_Capita_Income: float
    Median_Rent: float
    Median_Home_Value: float
    Unemployed_Population: float
    Latitude: float
    Longitude: float
    month: int
    quarter: int
    zipcode_freq: int
    cityTE: float

# CREATE FASTAPI OBJECT
app = FastAPI()


# PREDICT WITH FASTAPI
@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.dict()])
    df.columns = EXPECTED_COLUMNS
    
    pred = model.predict(df)[0]
    return {round(float(pred),2)}