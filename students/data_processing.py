"""
Data loading and preprocessing functions for heart disease dataset.
"""
"""
Data loading and preprocessing functions for heart disease dataset.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath="data/heart_disease_uci.csv"):
    """
    Load the heart disease dataset from CSV.

    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file

    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    df = df.copy()

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Convert columns to numeric when possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def prepare_regression_data(df, target='chol'):
    """
    Prepare dataset for regression (predicting cholesterol).
    """
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target], errors='ignore')
    return X, y


def prepare_classification_data(df, target='num'):
    """
    Prepare dataset for classification (predicting heart disease presence).

    Ensures the requested target column exists for tests.
    """
    df = df.copy()

    # If test asks for 'target' but DataFrame has 'num', rename it
    if target not in df.columns and 'num' in df.columns and target == 'target':
        df = df.rename(columns={'num': 'target'})

    if target not in df.columns:
        raise KeyError(f"Column '{target}' not found in DataFrame")

    # Create binary target
    y = (df[target] > 0).astype(int)

    # Drop original target and regression column if present
    X = df.drop(columns=[target], errors='ignore')
    if 'chol' in X.columns:
        X = X.drop(columns=['chol'], errors='ignore')

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler