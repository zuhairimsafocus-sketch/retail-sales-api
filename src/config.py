from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / 'sales_prediction_model_rf.pkl'

FEATURES = ['unit_price', 'quantity', 'brand', 'category']
TARGET = 'sales_amount'

VALID_BRANDS = ['Brand 1', 'Brand 2', 'Brand 3']
VALID_CATEGORIES = ['Beauty', 'Books', 'Clothing',
                    'Electronics', 'Groceries', 'Home', 'Sports', 'Toys']