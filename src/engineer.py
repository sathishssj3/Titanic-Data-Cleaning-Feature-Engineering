import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def engineer_features(df):
    """
    Performs feature engineering on the cleaned Titanic dataset.
    - Extracts titles from names
    - Creates age categories
    - Converts categorical features into numerical
    - Scales numerical features
    """
    print("\n--- 3. Feature Engineering ---")
    
    # 1. Extract titles from names (e.g. "Braund, Mr. Owen Harris" -> "Mr")
    print("Extracting titles from Names...")
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Consolidate rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Create age categories (Binning)
    print("Creating Age Categories...")
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child', 'Teenager', 'Young_Adult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Drop original Name and Ticket since they are text heavy and we extracted info from Name
    df = df.drop(columns=['Name', 'Ticket'])
    
    # 3. Convert categorical features to numerical
    print("Converting Categorical to Numerical (Encoding)...")
    
    # Label Encoding for binary or ordinal
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex']) # male/female to 1/0
    
    # One-Hot Encoding for nominal categories
    df = pd.get_dummies(df, columns=['Embarked', 'Title', 'AgeGroup'], drop_first=True)
    
    # 4. Feature scaling or normalization
    print("Scaling Numerical Features (Fare and Age)...")
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    print("\nFinal engineered dataset shape:", df.shape)
    print("\nFirst 5 rows of engineered dataset:")
    print(df.head())
    
    return df
