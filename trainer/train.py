import os
import argparse
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime


def load_data_from_bigquery(bq_table):
    """Load training data from BigQuery"""
    client = bigquery.Client()
    query = f"SELECT * FROM `{bq_table}`"
    df = client.query(query).to_dataframe()
    print(f"✓ Loaded {len(df)} rows from {bq_table}")
    return df


def preprocess_data(df):
    """Preprocess features and target"""
    # Separate features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Handle missing values
    X = X.fillna(X.mean())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'alpha': 0.5,
    }

    # Training with early stopping
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=10
    )

    return model, evals_result


def save_model(model, model_path, scaler, feature_names):
    """Save model and preprocessing artifacts to GCS"""

    # Save XGBoost model
    local_model_path = '/tmp/model.ubj'
    model.save_model(local_model_path)

    # Save scaler
    scaler_path = '/tmp/scaler.pkl'
    joblib.dump(scaler, scaler_path)

    # Save feature names
    features_path = '/tmp/features.json'
    with open(features_path, 'w') as f:
        json.dump({'features': list(feature_names)}, f)

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(model_path.split('/')[2])

    for local_file, gcs_file in [
        (local_model_path, f"{model_path}/model.ubj"),
        (scaler_path, f"{model_path}/scaler.pkl"),
        (features_path, f"{model_path}/features.json")
    ]:
        blob = bucket.blob(gcs_file.split(bucket.name + '/')[-1])
        blob.upload_from_filename(local_file)

    print(f"✓ Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bq_table', required=True, help='BigQuery table path')
    parser.add_argument('--model_output_path', required=True, help='GCS path for model output')
    args = parser.parse_args()

    # Load and preprocess
    df = load_data_from_bigquery(args.bq_table)
    X, y, scaler, feature_names = preprocess_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("\n🚀 Starting XGBoost training...")
    model, evals_result = train_xgboost_model(X_train, y_train, X_test, y_test)

    # Save model + artifacts
    save_model(model, args.model_output_path, scaler, feature_names)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
