import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, List

class DataTransformer:
    """
    Class to handle data transformation operations. This includes imputing missing values,
    standardizing and normalizing numerical data, and detecting outliers.
    """

    @staticmethod
    def convert_int_to_float(X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts integer columns to float type in the DataFrame.

        Parameters:
        - X (pd.DataFrame): Input DataFrame containing integer columns.

        Returns:
        - pd.DataFrame: Modified DataFrame with integer columns converted to float.
        """
        # Select integer columns
        int_cols = X.select_dtypes(include=['int']).columns
        # Convert integer columns to float
        X[int_cols] = X[int_cols].astype(float)
        return X

    @staticmethod
    def clean_NaN(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans NaN values in categorical columns by replacing them with 'NaN' string.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing categorical columns.

        Returns:
        - pd.DataFrame: Modified DataFrame with NaN values replaced by 'NaN' in categorical columns.
        """
        # Select categorical columns (object dtype)
        object_cols = df.select_dtypes(include=['object']).columns
        # Fill NaN values in categorical columns with 'NaN'
        df[object_cols] = df[object_cols].fillna('NaN')
        return df

    @staticmethod
    def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in numerical columns using KNNImputer.

        Parameters:
        - X (pd.DataFrame): Input DataFrame containing numerical columns with missing values.

        Returns:
        - pd.DataFrame: Modified DataFrame with imputed missing values using KNNImputer.
        """
        # Select numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            # Apply KNN imputation on numeric columns
            imputer = KNNImputer()
            imputed_values = imputer.fit_transform(X[numeric_cols])
            X[numeric_cols] = imputed_values
        return X

    @staticmethod
    def impute_missing_values_with_median(X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in numerical columns using the median of each column.

        Parameters:
        - X (pd.DataFrame): Input DataFrame containing numerical columns with missing values.

        Returns:
        - pd.DataFrame: Modified DataFrame with imputed missing values using the column median.
        """
        # Select numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            # Fill NaN values in numeric columns with the column median
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        return X

    @staticmethod
    def standardize_numeric_data(X: pd.DataFrame, ignore_columns: Optional[List[str]] = None) -> (pd.DataFrame, StandardScaler):
        """
        Standardizes numeric columns, excluding specified columns, using StandardScaler.

        Parameters:
        - X (pd.DataFrame): Input DataFrame containing numerical data.
        - ignore_columns (Optional[List[str]]): List of column names to exclude from standardization.

        Returns:
        - pd.DataFrame: Standardized DataFrame.
        - StandardScaler: Fitted scaler to reverse the transformation later if needed.
        """
        if ignore_columns is None:
            ignore_columns = []
        # Select numeric columns excluding the specified ones
        numeric_cols = X.select_dtypes(include=[float, int]).columns.difference(ignore_columns)
        scaler = None
        if not numeric_cols.empty:
            # Apply StandardScaler on the selected numeric columns
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X, scaler

    @staticmethod
    def normalize_numeric_data(X: pd.DataFrame, ignore_columns: Optional[List[str]] = None) -> (pd.DataFrame, MinMaxScaler):
        """
        Normalizes numeric columns, excluding specified columns, using MinMaxScaler.

        Parameters:
        - X (pd.DataFrame): Input DataFrame containing numerical data.
        - ignore_columns (Optional[List[str]]): List of column names to exclude from normalization.

        Returns:
        - pd.DataFrame: Normalized DataFrame.
        - MinMaxScaler: Fitted normalizer to reverse the transformation later if needed.
        """
        if ignore_columns is None:
            ignore_columns = []
        # Select numeric columns excluding the specified ones
        numeric_cols = X.select_dtypes(include=[float, int]).columns.difference(ignore_columns)
        normalizer = None
        if not numeric_cols.empty:
            # Apply MinMaxScaler on the selected numeric columns
            normalizer = MinMaxScaler()
            X[numeric_cols] = normalizer.fit_transform(X[numeric_cols])
        return X, normalizer

    @staticmethod
    def reverse_standardize(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """
        Reverses the standardization applied to numeric columns using the original scaler.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with standardized numerical columns.
        - scaler (StandardScaler): Fitted StandardScaler used to revert the standardization.

        Returns:
        - pd.DataFrame: DataFrame with reversed standardization.
        """
        # Select numeric columns
        numeric_cols = X.select_dtypes(include=[float, int]).columns
        # Reverse the standardization using the provided scaler
        X[numeric_cols] = scaler.inverse_transform(X[numeric_cols])
        return X

    @staticmethod
    def reverse_normalize(X: pd.DataFrame, normalizer: MinMaxScaler) -> pd.DataFrame:
        """
        Reverses the normalization applied to numeric columns using the original normalizer.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with normalized numerical columns.
        - normalizer (MinMaxScaler): Fitted MinMaxScaler used to revert the normalization.

        Returns:
        - pd.DataFrame: DataFrame with reversed normalization.
        """
        # Select numeric columns
        numeric_cols = X.select_dtypes(include=[float, int]).columns
        # Reverse the normalization using the provided normalizer
        X[numeric_cols] = normalizer.inverse_transform(X[numeric_cols])
        return X
