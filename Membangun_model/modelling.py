import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. KONFIGURASI KREDENSIAL DAGSHUB (SUDAH DISESUAIKAN)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'pitriabais'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '70059272253e0f09e6d4e535f41a5ebf493db8ff'
mlflow.set_tracking_uri('https://dagshub.com/pitriabais/Eksperimen_SML_Pitria-Bais-Mawarti.mlflow')

# Set Nama Eksperimen
mlflow.set_experiment("Heart_Disease_Modelling_Pitria")

def main():
    print("[INFO] Membaca dataset preprocessing...")
    train_df = pd.read_csv("heartdisease_preprocessing/train_clean.csv")
    test_df = pd.read_csv("heartdisease_preprocessing/test_clean.csv")
    
    # Memisahkan fitur dan target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    # Beri nama run di MLflow
    with mlflow.start_run(run_name="Baseline_RandomForest"):
        print("[INFO] Melatih Baseline Model...")
        
        # Hyperparameter dasar
        n_estimators = 100
        max_depth = 5
        random_state = 42
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"[RESULT] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
        # 2. MANUAL LOGGING PARAMETER & METRIK
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 3. MEMBUAT & MENYIMPAN 2 ARTEFAK TAMBAHAN
        # Artefak 1: Classification Report (File Teks)
        report_txt = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report_txt)
        mlflow.log_artifact("classification_report.txt") 
        
        # Artefak 2: Confusion Matrix (File Gambar Plot)
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Baseline')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png") 
        
        # Log Model
        mlflow.sklearn.log_model(model, "baseline_model")
        
        print("[SUCCESS] Baseline Model berhasil dicatat di DagsHub!")

if __name__ == "__main__":
    main()