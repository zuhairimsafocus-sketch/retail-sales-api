import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

FEATURES = ['unit_price', 'quantity', 'brand', 'category']
TARGET = 'sales_amount'

def get_preprocessor():
    return ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['brand', 'category'])
    ], remainder='passthrough')