from src.data_loader import load_data
from src.cleaner import clean_data
from src.engineer import engineer_features
import os

def main():
    print("======================================================")
    print(" Titanic Dataset: Data Cleaning & Feature Engineering ")
    print("======================================================")
    
    # 1. Load Data
    df = load_data()
    
    # 2. Clean Data
    df_clean = clean_data(df)
    
    # 3. Feature Engineering
    df_engineered = engineer_features(df_clean)
    
    # Optional: Save the final dataset
    os.makedirs('data', exist_ok=True)
    final_path = os.path.join('data', 'titanic_engineered.csv')
    df_engineered.to_csv(final_path, index=False)
    print(f"\nSuccessfully saved fully engineered dataset to: {final_path}")
    print("Ready for machine learning model training!")

if __name__ == "__main__":
    main()
