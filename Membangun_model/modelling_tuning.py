import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. KONFIGURASI KREDENSIAL DAGSHUB (SUDAH DISESUAIKAN)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'pitriabais'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '70059272253e0f09e6d4e535f41a5ebf493db8ff'
mlflow.set_tracking_uri('https://dagshub.com/pitriabais/Eksperimen_SML_Pitria-Bais-Mawarti.mlflow')

# Set Nama Eksperimen yang sama agar berkumpul dalam satu dashboard
mlflow.set_experiment("Heart_Disease_Modelling_Pitria")

def main():
    print("[INFO] Membaca dataset untuk Tuning...")
    train_df = pd.read_csv("heartdisease_preprocessing/train_clean.csv")
    test_df = pd.read_csv("heartdisease_preprocessing/test_clean.csv")
    
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    with mlflow.start_run(run_name="Tuned_RandomForest"):
        print("[INFO] Menjalankan Hyperparameter Tuning (GridSearchCV)...")
        
        # Tentukan kandidat parameter yang dituning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7]
        }
        
        base_rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='f1_macro')
        grid_search.fit(X_train, y_train)
        
        # Ambil model terbaik hasil tuning
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"[RESULT TUNED] Best Params: {grid_search.best_params_}")
        print(f"[RESULT TUNED] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
        # 2. MANUAL LOGGING UNTUK TUNING
        mlflow.log_param("model_type", "RandomForest_Tuned")
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 3. MEMBUAT & MENYIMPAN 2 ARTEFAK TAMBAHAN TUNED
        # ARTEFAK TAMBAHAN 1: Classification Report Tuned
        report_txt = classification_report(y_test, y_pred)
        with open("classification_report_tuned.txt", "w") as f:
            f.write(report_txt)
        mlflow.log_artifact("classification_report_tuned.txt")
        
        # ARTEFAK TAMBAHAN 2: Confusion Matrix Tuned (Warna Ungu)
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
        plt.title('Confusion Matrix - Tuned Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix_tuned.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix_tuned.png")
        
        # Log Best Model
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        print("[SUCCESS] Tuned Model berhasil dicatat di DagsHub!")

if __name__ == "__main__":
    main()