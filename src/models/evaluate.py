import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from src.config import MODEL_PATH, FEATURES, TARGET

def evaluate(data_path: str):
    df = pd.read_csv(data_path)
    df.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name',
                     'transaction_date'], inplace=True, errors='ignore')

    X = df[FEATURES]
    y = df[TARGET]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring='r2')

    print("=" * 45)
    print("  MODEL EVALUATION REPORT")
    print("=" * 45)
    print(f"  R²        : {r2:.4f}")
    print(f"  RMSE      : {rmse:.2f}")
    print(f"  MAE       : {mae:.2f}")
    print(f"  CV Mean   : {cv.mean():.4f}")
    print(f"  CV Std    : {cv.std():.4f}")
    print("=" * 45)
    return {"r2": r2, "rmse": rmse, "mae": mae,
            "cv_mean": cv.mean(), "cv_std": cv.std()}

if __name__ == "__main__":
    evaluate("retail_sales_dataset.csv")