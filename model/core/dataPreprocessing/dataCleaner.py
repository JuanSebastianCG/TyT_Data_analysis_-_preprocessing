import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, List
from .encoders import Encoder

# Main DataCleaner class
class DataCleaner:
    """
    A class to handle data cleaning operations. This includes imputing missing values, standardizing and normalizing
    numeric data, encoding categorical variables, and detecting outliers.
    """

    def __init__(self):
        self.encoder = Encoder()  # Initialize the encoder class for encoding categorical variables

    @staticmethod
    def convert_int_to_float(X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts integer columns to float in the DataFrame.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: The DataFrame with integer columns converted to float.
        """
        int_cols = X.select_dtypes(include=['int']).columns
        X[int_cols] = X[int_cols].astype(float)
        return X

    @staticmethod
    def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in numeric columns using KNNImputer.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: The DataFrame with imputed missing values.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            imputer = KNNImputer()
            imputed_values = imputer.fit_transform(X[numeric_cols])
            X[numeric_cols] = imputed_values
        return X

    @staticmethod
    def impute_missing_values_with_median(X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in numeric columns with the median of the column.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: The DataFrame with missing values imputed by the median.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        return X

    @staticmethod
    def standardize_numeric_data(X: pd.DataFrame, ignore_columns: Optional[List[str]] = None) -> (pd.DataFrame, StandardScaler):
        """
        Standardizes numeric columns by scaling them to have zero mean and unit variance, excluding specified columns.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            ignore_columns (Optional[List[str]]): List of columns to ignore from standardization. Default is None.
        
        Returns:
            pd.DataFrame: The standardized DataFrame.
            StandardScaler: The scaler object used to revert the standardization.
        """
        if ignore_columns is None:
            ignore_columns = []
        numeric_cols = X.select_dtypes(include=[float, int]).columns.difference(ignore_columns)
        if not numeric_cols.empty:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X, scaler

    @staticmethod
    def normalize_numeric_data(X: pd.DataFrame, ignore_columns: Optional[List[str]] = None) -> (pd.DataFrame, MinMaxScaler):
        """
        Normalizes numeric columns by scaling them to a range between 0 and 1, excluding specified columns.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            ignore_columns (Optional[List[str]]): List of columns to ignore from normalization. Default is None.
        
        Returns:
            pd.DataFrame: The normalized DataFrame.
            MinMaxScaler: The normalizer object used to revert the normalization.
        """
        if ignore_columns is None:
            ignore_columns = []
        numeric_cols = X.select_dtypes(include=[float, int]).columns.difference(ignore_columns)
        if not numeric_cols.empty:
            normalizer = MinMaxScaler()
            X[numeric_cols] = normalizer.fit_transform(X[numeric_cols])
        return X, normalizer

    @staticmethod
    def reverse_standardize(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """
        Reverts the standardization applied to numeric columns.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            scaler (StandardScaler): The scaler object used for standardization.
        
        Returns:
            pd.DataFrame: The DataFrame with reverted standardization.
        """
        numeric_cols = X.select_dtypes(include=[float, int]).columns
        X[numeric_cols] = scaler.inverse_transform(X[numeric_cols])
        return X

    @staticmethod
    def reverse_normalize(X: pd.DataFrame, normalizer: MinMaxScaler) -> pd.DataFrame:
        """
        Reverts the normalization applied to numeric columns.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            normalizer (MinMaxScaler): The normalizer object used for normalization.
        
        Returns:
            pd.DataFrame: The DataFrame with reverted normalization.
        """
        numeric_cols = X.select_dtypes(include=[float, int]).columns
        X[numeric_cols] = normalizer.inverse_transform(X[numeric_cols])
        return X

    @staticmethod
    def apply_encoding(X: pd.DataFrame, encoding_method: str = 'label', columns: Optional[List[str]] = None) -> (pd.DataFrame, Encoder):
        """
        Applies a specified encoding method (Label, One-Hot, or Frequency) to the specified columns.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            encoding_method (str): The encoding method to apply ('label', 'onehot', 'frequency'). Default is 'label'.
            columns (Optional[List[str]]): The columns to encode. If None, all object-type columns will be encoded.
        
        Returns:
            pd.DataFrame: The encoded DataFrame.
            Encoder: The encoder object used to revert the encoding.
        """
        encoder = Encoder()
        if columns is None:
            columns = X.select_dtypes(include=['object']).columns.tolist()
        if encoding_method == 'label':
            X = encoder.label_encode(X, columns)
        elif encoding_method == 'onehot':
            X = encoder.one_hot_encode(X, columns)
        elif encoding_method == 'frequency':
            X = encoder.frequency_encode(X, columns)
        return X, encoder

    @staticmethod
    def reverse_encoding(X: pd.DataFrame, encoding_method: str, encoder: Encoder, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reverts the encoding applied to the specified columns back to their original categorical values.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            encoding_method (str): The encoding method to revert ('label', 'onehot', 'frequency').
            encoder (Encoder): The encoder object used for encoding.
            columns (Optional[List[str]]): The columns to decode. If None, it will use the last encoded columns.
        
        Returns:
            pd.DataFrame: The DataFrame with reverted encodings.
        """
        if columns is None:
            columns = encoder.last_features_encoded
        if encoding_method == 'label':
            X = encoder.reverse_label_encode(X, columns)
        elif encoding_method == 'onehot':
            X = encoder.reverse_one_hot_encode(X, columns)
        elif encoding_method == 'frequency':
            X = encoder.reverse_frequency_encode(X, columns)
        return X
