import pytest
import pandas as pd
from model.predict import load_model, predict
from src.config import MODEL_PATH, VALID_BRANDS, VALID_CATEGORIES

@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)

def test_model_loads(model):
    assert model is not None

def test_predict_returns_float(model):
    result = predict(model, unit_price=300.0, quantity=2,
                     brand="Brand 1", category="Sports")
    assert isinstance(result, float)

def test_predict_positive_output(model):
    result = predict(model, unit_price=300.0, quantity=2,
                     brand="Brand 1", category="Sports")
    assert result > 0

def test_predict_higher_price_higher_sales(model):
    low  = predict(model, unit_price=50.0,  quantity=1, brand="Brand 1", category="Books")
    high = predict(model, unit_price=400.0, quantity=1, brand="Brand 1", category="Books")
    assert high > low

def test_predict_higher_quantity_higher_sales(model):
    q1 = predict(model, unit_price=200.0, quantity=1, brand="Brand 2", category="Home")
    q3 = predict(model, unit_price=200.0, quantity=3, brand="Brand 2", category="Home")
    assert q3 > q1

def test_predict_unknown_brand_no_crash(model):
    # OneHotEncoder handle_unknown='ignore' — tak patut crash
    result = predict(model, unit_price=200.0, quantity=1,
                     brand="Unknown Brand", category="Sports")
    assert isinstance(result, float)

def test_predict_unknown_category_no_crash(model):
    result = predict(model, unit_price=200.0, quantity=1,
                     brand="Brand 1", category="Unknown")
    assert isinstance(result, float)