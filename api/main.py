from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from model.predict import predict

app = FastAPI(title='Sales Prediction API')
model = joblib.load('sales_prediction_model_rf.pkl')

class PredictRequest(BaseModel):
    unit_price: float
    quantity: int
    brand: str
    category: str
    
class PredictResponse(BaseModel):
    predicted_sales_amount: float
    
@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictResponse)
def predict_sales(req: PredictRequest):
    result = predict(model, req.unit_price, req.quantity, req.brand, req.category)
    return PredictResponse(predicted_sales_amount=round(result, 2))

    