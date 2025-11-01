import pandas as pd
import joblib
import mlflow
import os
import argparse # Kita gunakan argparse untuk mengambil parameter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi-fungsi helper
def load_data(path):
    print(f"Memuat data dari {path}...")
    return pd.read_csv(path)

def split_data(df, target_col='Churn', test_size=0.2, random_state=42):
    print("Membagi data...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main():
    # Mengambil parameter yang diberikan oleh 'mlflow run'
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--data_path", type=str, default="data/telco_churn_preprocessed.csv")
    args = parser.parse_args()

    # Set nama eksperimen
    mlflow.set_experiment("Telco_Churn_CI_Workflow")

    with mlflow.start_run():
        print("Memulai MLflow Run...")
        
        # 1. Log Parameter (secara manual)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("data_path", args.data_path)

        # 2. Load & Split Data
        df = load_data(args.data_path)
        X_train, X_test, y_train, y_test = split_data(df)

        # 3. Latih Model
        print("Melatih model RandomForest...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 4. Evaluasi Model
        print("Mengevaluasi model...")
        y_pred = model.predict(X_test)
        
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred)
        }
        
        # 5. Log Metrik
        mlflow.log_metrics(metrics)
        
        # 6. Log Model (Artifact)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\nModel dilatih. Metrik: {metrics}")
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Workflow CI selesai.")

if __name__ == "__main__":
    main()