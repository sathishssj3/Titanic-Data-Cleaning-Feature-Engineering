import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = os.path.join("data", "titanic.csv")

def download_data():
    """Downloads the Titanic dataset if it doesn't exist locally."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(DATA_PATH):
        print("Downloading Titanic dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Dataset downloaded.")

def load_data():
    """Loads the dataset into a pandas DataFrame."""
    download_data()
    df = pd.read_csv(DATA_PATH)
    print("\n--- 1. Loading the Titanic Dataset ---")
    print(f"Dataset Shape: {df.shape}")
    return df
