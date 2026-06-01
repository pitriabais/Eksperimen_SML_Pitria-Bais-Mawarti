import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi Kredensial Otomatis ke DagsHub Pitria
os.environ['MLFLOW_TRACKING_USERNAME'] = 'pitriabais'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '70059272253e0f09e6d4e535f41a5ebf493db8ff'
mlflow.set_tracking_uri('https://dagshub.com/pitriabais/Eksperimen_SML_Pitria-Bais-Mawarti.mlflow')

mlflow.set_experiment("Heart_Disease_Modelling_Pitria")

def main():
    print("[INFO] Membaca dataset preprocessing lokal...")
    train_df = pd.read_csv("heartdisease_preprocessing/train_clean.csv")
    test_df = pd.read_csv("heartdisease_preprocessing/test_clean.csv")
    
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    
    # Memulai pencatatan otomatis alur CI
    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        print("[INFO] Melatih ulang model via GitHub Actions...")
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan parameter dasar
        mlflow.log_param("model_type", "RandomForest_CI")
        mlflow.log_param("status", "Retrained_Automatically")
        
        # Log model ke DagsHub agar dapet run_id untuk Docker Build
        mlflow.sklearn.log_model(model, "model")
        print("[SUCCESS] Model baru berhasil dikirim ke DagsHub!")

if __name__ == "__main__":
    main()