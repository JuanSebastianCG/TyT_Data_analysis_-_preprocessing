import pandas as pd
from typing import Optional, List, Dict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Encoder:
    """
    Class for handling categorical variable encodings, including One-Hot Encoding, Label Encoding, and Frequency Encoding.
    """

    def __init__(self):
        """
        Initializes the Encoder class by creating dictionaries to store LabelEncoders, OneHotEncoders, and Frequency Encoding mappings.
        Also stores the last set of encoded features for potential reverse operations.
        """
        self.label_encoders: Dict[str, LabelEncoder] = {}  # Stores label encoders for each column
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}  # Stores one-hot encoders for each column
        self.frequency_encodings: Dict[str, Dict] = {}  # Stores frequency encoding mappings
        self.last_features_encoded = []  # Tracks the most recent set of encoded columns

    def one_hot_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies One-Hot Encoding to specified columns.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - columns (Optional[List[str]]): List of column names to one-hot encode. Defaults to all categorical columns.

        Returns:
        - pd.DataFrame: DataFrame with one-hot encoded columns.
        """
        if columns is None:
            # If no columns are specified, select all object-type columns
            columns = df.select_dtypes(include=['object']).columns.tolist()

        df = df.copy()  # Explicitly copy the DataFrame to avoid warning on modifications
        for col in columns:
            df[col] = df[col].fillna('NaN').astype(str)  # Fill missing values with 'NaN' string

        # Initialize and fit OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(
            encoder.fit_transform(df[columns]),  # Apply One-Hot encoding
            columns=encoder.get_feature_names_out(),  # Generate feature names
            index=df.index  # Keep original index
        )

        # Drop original columns and join encoded columns to the DataFrame
        df = df.drop(columns=columns).join(encoded_data)
        self.onehot_encoders[col] = encoder  # Save encoder for potential reversal later
        self.last_features_encoded = columns  # Store the encoded columns
        return df

    def label_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies Label Encoding to specified columns.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - columns (Optional[List[str]]): List of column names to label encode. Defaults to all categorical columns.

        Returns:
        - pd.DataFrame: DataFrame with label encoded columns.
        """
        if columns is None:
            # If no columns are specified, select all object-type columns
            columns = df.select_dtypes(include=['object']).columns.tolist()

        df = df.copy()  # Explicitly copy the DataFrame
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('NaN'))  # Apply Label Encoding and fill NaN
            self.label_encoders[col] = le  # Save encoder for potential reversal
            self.last_features_encoded = columns  # Store the encoded columns
        return df

    def frequency_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies Frequency Encoding to specified columns.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - columns (Optional[List[str]]): List of column names to frequency encode. Defaults to all categorical columns.

        Returns:
        - pd.DataFrame: DataFrame with frequency encoded columns.
        """
        if columns is None:
            # If no columns are specified, select all object-type columns
            columns = df.select_dtypes(include=['object']).columns.tolist()

        df = df.copy()  # Explicitly copy the DataFrame
        for col in columns:
            freq_encoding = df[col].value_counts() / len(df)  # Calculate frequency for each value
            df[col] = df[col].map(freq_encoding).fillna(0)  # Map frequencies to the column
            self.frequency_encodings[col] = freq_encoding.to_dict()  # Save encoding mapping
            self.last_features_encoded = columns  # Store the encoded columns
        return df

    def reverse_label_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reverts Label Encoding on specified columns back to their original values.

        Parameters:
        - df (pd.DataFrame): Input DataFrame with label encoded columns.
        - columns (Optional[List[str]]): List of column names to decode. Defaults to the last encoded columns.

        Returns:
        - pd.DataFrame: DataFrame with label decoded columns.
        """
        if columns is None:
            # Use last encoded columns if none are specified
            columns = self.last_features_encoded

        if columns is None or not columns:
            raise ValueError("No columns specified for decoding, and no previously encoded columns available.")

        df = df.copy()  # Explicitly copy the DataFrame
        for col in columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                decoded_values = le.inverse_transform(df[col].astype(int))  # Apply inverse transformation
                df[col] = pd.Series(decoded_values, dtype='object', index=df.index)  # Restore original values
        return df

    def reverse_one_hot_encode(self, df: pd.DataFrame, original_columns: List[str]) -> pd.DataFrame:
        """
        Reverts One-Hot Encoding on specified columns back to their original values.

        Parameters:
        - df (pd.DataFrame): Input DataFrame with one-hot encoded columns.
        - original_columns (List[str]): List of original columns that were one-hot encoded.

        Returns:
        - pd.DataFrame: DataFrame with one-hot decoded columns.
        """
        df = df.copy()  # Explicitly copy the DataFrame
        for col in original_columns:
            if col in self.onehot_encoders:
                ohe = self.onehot_encoders[col]
                ohe_columns = ohe.get_feature_names_out([col])  # Get encoded column names

                # Check if all one-hot columns are present in the DataFrame
                ohe_columns_in_df = [c for c in ohe_columns if c in df.columns]

                if not ohe_columns_in_df:
                    raise ValueError(f"No one-hot encoded columns for '{col}' found in the DataFrame.")

                # Revert one-hot encoded columns back to original values
                original_values = ohe.inverse_transform(df[ohe_columns_in_df])
                df[col] = original_values[:, 0]  # Select the first column since it's 2D

                # Drop one-hot encoded columns from the DataFrame
                df = df.drop(columns=ohe_columns_in_df)
        return df

    def reverse_frequency_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reverts Frequency Encoding on specified columns back to their original values.

        Parameters:
        - df (pd.DataFrame): Input DataFrame with frequency encoded columns.
        - columns (Optional[List[str]]): List of column names to decode. Defaults to the last encoded columns.

        Returns:
        - pd.DataFrame: DataFrame with frequency decoded columns.
        """
        if columns is None:
            # Use last encoded columns if none are specified
            columns = self.last_features_encoded

        if not columns:
            raise ValueError("No columns specified for decoding, and no previously encoded columns available.")

        df = df.copy()  # Explicitly copy the DataFrame
        for col in columns:
            if col in self.frequency_encodings:
                # Reverse the frequency encoding by mapping the frequency values back to the original categories
                reverse_mapping = {v: k for k, v in self.frequency_encodings[col].items()}
                df[col] = df[col].map(reverse_mapping).fillna('Unknown')  # Restore original values
        return df
