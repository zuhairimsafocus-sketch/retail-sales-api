import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from src.config import MODEL_PATH, FEATURES, TARGET

def build_pipeline():
    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['brand', 'category'])
    ], remainder='passthrough')

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

def train(data_path: str):
    df = pd.read_csv(data_path)
    df.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name',
                     'transaction_date'], inplace=True)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("retail-sales-prediction")

    with mlflow.start_run(run_name="random_forest_retrain"):
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        cv   = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("features", str(FEATURES))
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("rmse_test", rmse)
        mlflow.log_metric("mae_test", mae)
        mlflow.log_metric("cv_mean_r2", cv.mean())
        mlflow.log_metric("cv_std_r2", cv.std())

        mlflow.sklearn.log_model(pipeline, "model_rf",
                                  registered_model_name="retail-sales-rf")
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Model saved → {MODEL_PATH}")
        print(f"R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

if __name__ == "__main__":
    train("retail_sales_dataset.csv")