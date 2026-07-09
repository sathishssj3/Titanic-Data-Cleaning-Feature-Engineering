import pandas as pd

def clean_data(df):
    """
    Cleans the Titanic dataset by handling missing values, 
    removing duplicates, and correcting data types.
    """
    print("\n--- 2. Data Cleaning ---")
    
    # Check initial missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # 1. Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} duplicate rows.")
    
    # 2. Handle missing values
    # Age: Fill missing with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Fill missing with the most frequent value (mode)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Too many missing values, often better to drop or convert to a binary "Has_Cabin"
    # For this exercise, let's drop the Cabin column as it's highly sparse
    df = df.drop(columns=['Cabin'])
    
    # Fare: Fill missing with median (just in case there are any)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum().sum(), "total missing values.")
    
    # 3. Correct inconsistent data types
    # Ensure passenger ID, survived, pclass are integers
    df['PassengerId'] = df['PassengerId'].astype(int)
    df['Survived'] = df['Survived'].astype(int)
    df['Pclass'] = df['Pclass'].astype(int)
    
    return df
