import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_automation():
    print("[INFO] Memulai pipeline preprocessing otomatis...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, 'heartdisease_raw', 'heart_disease.csv')
    output_dir = os.path.join(base_dir, 'preprocessing', 'heartdisease_preprocessing')
    
    if not os.path.exists(raw_data_path):
        print(f"[ERROR] File mentah tidak ditemukan di: {raw_data_path}")
        sys.exit(1)
        
    df = pd.read_csv(raw_data_path)
    df = df.drop_duplicates().dropna()
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_clean = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_clean['target'] = y_train.values
    
    test_clean = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_clean['target'] = y_test.values
    
    os.makedirs(output_dir, exist_ok=True)
    train_clean.to_csv(os.path.join(output_dir, 'train_clean.csv'), index=False)
    test_clean.to_csv(os.path.join(output_dir, 'test_clean.csv'), index=False)
    print(f"[SUCCESS] Pipeline selesai! File disimpan di: {output_dir}")

if __name__ == "__main__":
    run_automation()