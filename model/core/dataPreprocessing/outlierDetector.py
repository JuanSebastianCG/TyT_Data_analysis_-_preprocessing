import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Class OutlierDetector to handle outliers and track which columns caused the removal
class OutlierDetector:
    """
    Class to handle the detection and removal of outliers using different methods.
    Allows tracking of which specific columns caused the removal of each row.
    """

    def __init__(self):
        """
        Initializes the OutlierDetector class with two DataFrames:
        - removed_outliers: Stores the rows that were detected as outliers.
        - removal_reasons: Stores the columns that caused the removal of the corresponding outlier rows.
        """
        self.removed_outliers = pd.DataFrame()  # Stores outlier rows
        self.removal_reasons = pd.DataFrame()  # Stores reasons for removal (columns responsible)

    def detect_and_remove_outliers(self, X: pd.DataFrame, method: str = 'zscore', ignore_columns: list = None, **kwargs) -> pd.DataFrame:
        """
        Detects and removes outliers using the specified method, while ignoring the specified columns.
        Saves the removed data and tracks the columns that caused the removal.

        Args:
            X (pd.DataFrame): The DataFrame with data to analyze.
            method (str): The outlier detection method to use ('zscore', 'iqr', 'isolation_forest').
            ignore_columns (list): List of columns to ignore during outlier detection.
            kwargs: Additional parameters for the detection methods.

        Returns:
            pd.DataFrame: A cleaned DataFrame without the outliers.
        """
        if ignore_columns is None:
            ignore_columns = []  # Initialize with an empty list if no columns are provided

        # Exclude columns to ignore from the analysis
        X_to_analyze = X.drop(columns=ignore_columns, errors='ignore')

        # Choose the outlier detection method
        if method == 'zscore':
            self.removed_outliers, self.removal_reasons = self._detect_outliers_zscore(X, X_to_analyze, **kwargs)
        elif method == 'iqr':
            self.removed_outliers, self.removal_reasons = self._detect_outliers_iqr(X, X_to_analyze, **kwargs)
        elif method == 'isolation_forest':
            self.removed_outliers, self.removal_reasons = self._detect_outliers_isolation_forest(X, X_to_analyze, **kwargs)

        # Remove outliers from the DataFrame and return the cleaned DataFrame
        X_cleaned = X.drop(self.removed_outliers.index)
        return X_cleaned

    def _detect_outliers_zscore(self, X: pd.DataFrame, X_to_analyze: pd.DataFrame, threshold: float = 3.0) -> (pd.DataFrame, pd.DataFrame):
        """
        Detects and removes outliers using the Z-score method and tracks which column caused the removal.

        Args:
            X (pd.DataFrame): The original DataFrame.
            X_to_analyze (pd.DataFrame): The subset of the DataFrame for outlier detection.
            threshold (float): Z-score threshold for defining outliers.

        Returns:
            removed_outliers (pd.DataFrame): Rows detected as outliers.
            removal_reasons_df (pd.DataFrame): DataFrame storing the column responsible for each outlier.
        """
        # Select numeric columns
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        # Calculate Z-scores
        z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0))
        # Identify rows with any Z-score above the threshold
        outliers = (z_scores > threshold).any(axis=1)
        removed_outliers = X[outliers]  # Store rows detected as outliers
        # Track the column responsible for the highest Z-score
        removal_reasons = z_scores[outliers].idxmax(axis=1)
        removal_reasons_df = pd.DataFrame({'Column': removal_reasons})
        return removed_outliers, removal_reasons_df

    def _detect_outliers_iqr(self, X: pd.DataFrame, X_to_analyze: pd.DataFrame, multiplier: float = 1.5) -> (pd.DataFrame, pd.DataFrame):
        """
        Detects and removes outliers using the Interquartile Range (IQR) method and tracks which column caused the removal.

        Args:
            X (pd.DataFrame): The original DataFrame.
            X_to_analyze (pd.DataFrame): The subset of the DataFrame for outlier detection.
            multiplier (float): Multiplier for the IQR range to define outliers.

        Returns:
            removed_outliers (pd.DataFrame): Rows detected as outliers.
            removal_reasons_df (pd.DataFrame): DataFrame storing the column responsible for each outlier.
        """
        # Select numeric columns
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        # Calculate the first and third quartiles (Q1 and Q3)
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1  # Interquartile Range
        # Define outliers as values outside the IQR bounds
        is_outlier = ((numeric_cols < (Q1 - multiplier * IQR)) | (numeric_cols > (Q3 + multiplier * IQR)))
        outlier_rows = is_outlier.any(axis=1)
        removed_outliers = X[outlier_rows]  # Store rows detected as outliers
        # Track the column responsible for the outlier
        removal_reasons = is_outlier[outlier_rows].idxmax(axis=1)
        removal_reasons_df = pd.DataFrame({'Column': removal_reasons})
        return removed_outliers, removal_reasons_df

    def _detect_outliers_isolation_forest(self, X: pd.DataFrame, X_to_analyze: pd.DataFrame, contamination: float = 0.01) -> (pd.DataFrame, pd.DataFrame):
        """
        Detects and removes outliers using Isolation Forest. Does not track the specific column responsible for the removal.

        Args:
            X (pd.DataFrame): The original DataFrame.
            X_to_analyze (pd.DataFrame): The subset of the DataFrame for outlier detection.
            contamination (float): The proportion of data points to consider outliers.

        Returns:
            removed_outliers (pd.DataFrame): Rows detected as outliers.
            removal_reasons_df (pd.DataFrame): DataFrame indicating multiple columns were responsible for each outlier.
        """
        # Select numeric columns
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        # Apply Isolation Forest algorithm
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(numeric_cols)
        removed_outliers = X[outlier_pred == -1]  # Store rows detected as outliers
        # Since Isolation Forest considers all columns, mark the reason as 'Multiple'
        removal_reasons_df = pd.DataFrame({'Column': ['Multiple'] * len(removed_outliers)})
        return removed_outliers, removal_reasons_df

    def show_outliers_by_column(self, column_name: str) -> None:
        """
        Displays the outliers removed that are related to a specific column.

        Args:
            column_name (str): The name of the column to check for outliers.

        Returns:
            None
        """
        if column_name not in self.removal_reasons['Column'].values:
            print(f"No outliers found related to the column '{column_name}'.")
            return

        # Identify the rows that were outliers due to the specified column
        related_outliers = self.removal_reasons[self.removal_reasons['Column'] == column_name]
        related_outlier_rows = self.removed_outliers.loc[related_outliers.index]
        print(f"\nOutliers removed for the column '{column_name}':")
        print(related_outlier_rows[column_name])
