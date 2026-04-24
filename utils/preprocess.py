import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Columns that don’t really help with prediction or may leak information
# (especially churn-related ones like ChurnReason)
DROP_COLS = [
    "CustomerID", "Country", "State", "City", "ZipCode",
    "Latitude", "Longitude", "Quarter",
    "ChurnCategory", "ChurnReason", "ChurnScore", "CLTV",
    "CustomerStatus", "Population","SatisfactionScore"
]

def load_and_clean(filepath: str) -> pd.DataFrame:
    # Load dataset
    df = pd.read_csv(filepath)

    # Just in case there are unwanted spaces in column names
    df.columns = df.columns.str.strip()

    # Drop unnecessary / leakage columns (only if they exist)
    drop_cols = [col for col in DROP_COLS if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Some datasets use blank spaces instead of proper NaNs
    df.replace(" ", np.nan, inplace=True)

    # Dropping rows where too many values are missing
    # (keeping rows that have at least half the data)
    df.dropna(thresh=int(len(df.columns) * 0.5), inplace=True)
    
    # Handle meaningful missing values first
    if "InternetType" in df.columns:
        df["InternetType"] = df["InternetType"].fillna("No Internet")

    if "Offer" in df.columns:
        df["Offer"] = df["Offer"].fillna("No Offer")
    # Filling missing values:
    # - categorical → mode (most frequent)
    # - numerical → median (more robust than mean)
    # Drop fully empty categorical columns first
    empty_cols = [col for col in df.select_dtypes(include="object").columns if df[col].isnull().all()]
    df.drop(columns=empty_cols, inplace=True)

    # Then fill remaining
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
   

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def encode_and_scale(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    y = y.astype(str).str.lower().str.strip()
    y = y.isin(["yes", "1", "true", "churned"]).astype(int)

    cat_cols = X.select_dtypes(include="object").columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders


def build_pipeline(filepath: str, test_size=0.2, random_state=42):
    # Step 1: Load and clean data
    df = load_and_clean(filepath)

    def get_target_column(df):
        if "ChurnLabel" in df.columns:

            return "ChurnLabel"
        elif "Churn" in df.columns:
            return "Churn"
        else:
            raise ValueError("Target column not found")

    target_col = get_target_column(df)

    # Step 2: Encode only (no scaling yet)
    X, y, encoders = encode_and_scale(df, target_col)
    # Step 3: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Step 4: Scale AFTER split
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include="number").columns

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Quick sanity check
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    feature_names = X.columns.tolist()

    return X_test, X_train, y_test, y_train, scaler, encoders, feature_names, df, target_col
