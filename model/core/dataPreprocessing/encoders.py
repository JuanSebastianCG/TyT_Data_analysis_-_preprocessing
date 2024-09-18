import pandas as pd
from typing import Optional, List, Dict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Clase Encoder para codificar y decodificar variables categóricas
class Encoder:
    """
    Clase para manejar las codificaciones de variables categóricas (One-Hot, Label, Frequency).
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.frequency_encodings: Dict[str, Dict] = {}
        self.last_features_encoded = []

    def one_hot_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Aplica One-Hot Encoding a las columnas especificadas."""
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            df[col] = df[col].fillna('NaN').astype(str)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(
            encoder.fit_transform(df[columns]),
            columns=encoder.get_feature_names_out(),
            index=df.index
        )
        df = df.drop(columns=columns).join(encoded_data)
        self.onehot_encoders[col] = encoder  
        self.last_features_encoded = columns
        return df

    def label_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Aplica Label Encoding a las columnas especificadas."""
        for col in columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('NaN').astype(str)
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le  
            self.last_features_encoded = columns
        return df

    def frequency_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Aplica Frequency Encoding a las columnas especificadas."""
        for col in columns:
            freq_encoding = df[col].value_counts() / len(df)
            df[col] = df[col].map(freq_encoding).fillna(0)
            self.frequency_encodings[col] = freq_encoding.to_dict()  
            self.last_features_encoded = columns
        return df

    def reverse_label_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Decodifica las columnas con Label Encoding de vuelta a sus valores originales."""
        for col in columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = le.inverse_transform(df[col])
        return df

    def reverse_one_hot_encode(self, df: pd.DataFrame, original_columns: List[str]) -> pd.DataFrame:
        """Decodifica las columnas con One-Hot Encoding de vuelta a sus valores originales."""
        for col in original_columns:
            if col in self.onehot_encoders:
                ohe = self.onehot_encoders[col]
                # Obtener las columnas originales y revertir el proceso
                ohe_columns = ohe.get_feature_names_out([col])
                original_values = ohe.inverse_transform(df[ohe_columns])
                df[col] = original_values
                df = df.drop(columns=ohe_columns)
        return df

    def reverse_frequency_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Decodifica las columnas con Frequency Encoding de vuelta a sus valores originales."""
        for col in columns:
            if col in self.frequency_encodings:
                reverse_mapping = {v: k for k, v in self.frequency_encodings[col].items()}
                df[col] = df[col].map(reverse_mapping).fillna('Unknown')
        return df
