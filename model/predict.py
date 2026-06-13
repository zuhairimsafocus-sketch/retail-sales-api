import joblib
import pandas as pd
import numpy as np

def load_model(path: str):
    return joblib.load(path)

def predict(model, unit_price: float, quantity: int, brand: str, category: str):
    input_data = pd.DataFrame([{
        'unit_price' : unit_price,
        'quantity' : quantity,
        'brand' : brand, 
        'category' : category
    }])
    
    return float(model.predict(input_data)[0])