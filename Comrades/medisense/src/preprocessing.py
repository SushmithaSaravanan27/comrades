import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_data(file_path):
    """
    Load dataset
    """
    df = pd.read_csv(file_path)
    return df


def handle_missing_values(df):
    """
    Replace medically impossible zeros with NaN
    and create missing indicator flags
    """

    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in zero_columns:
        # Create missing flag
        df[f"{col}_Missing"] = (df[col] == 0).astype(int)

        # Replace 0 with NaN
        df[col] = df[col].replace(0, np.nan)

    return df


def impute_missing_values(df):
    """
    Apply median imputation
    """

    columns_to_impute = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    imputer = SimpleImputer(strategy="median")

    df[columns_to_impute] = imputer.fit_transform(
        df[columns_to_impute]
    )

    return df


def preprocess_data(file_path):
    """
    Full preprocessing pipeline
    """

    df = load_data(file_path)

    print("\n Dataset Loaded")
    print(df.shape)

    df = handle_missing_values(df)

    print("\n Missing Values Converted")

    df = impute_missing_values(df)

    print("\n Median Imputation Completed")

    print("\n Remaining Null Values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    df_clean = preprocess_data("data/diabetes.csv")

    print("\n First Five Rows:")
    print(df_clean.head())

    print("\n Dataset Shape:")
    print(df_clean.shape)

    