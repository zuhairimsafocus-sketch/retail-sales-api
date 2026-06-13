from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_valid():
    payload = {
        "unit_price": 300.0,
        "quantity": 2,
        "brand": "Brand 1",
        "category": "Sports"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_sales_amount" in response.json()
    assert response.json()["predicted_sales_amount"] > 0

def test_predict_endpoint_missing_field():
    payload = {"unit_price": 300.0, "quantity": 2}  # brand & category missing
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # FastAPI validation error