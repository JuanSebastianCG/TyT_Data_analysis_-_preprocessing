import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, List
from .encoders import Encoder

# Clase principal DataCleaner
class DataCleaner:
    """
    Clase para manejar la limpieza de datos. Coordina la imputación, estandarización, normalización, 
    codificación de variables categóricas y detección de outliers.
    """
    def __init__(self, X: pd.DataFrame, feature_selected: Optional[List[str]] = None):
        """
        Inicializa la clase DataCleaner con las características (X).
        Ofrece la opción de seleccionar características específicas.
        """
        self.X = X.copy()
        self.stored_scaler = None
        self.stored_normalizer = None
        self.encoder = Encoder()

    def convert_int_to_float(self) -> None:
        """Convierte columnas de tipo entero a flotante en el DataFrame."""
        int_cols = self.X.select_dtypes(include=['int']).columns
        self.X[int_cols] = self.X[int_cols].astype(float)

    def impute_missing_values(self) -> None:
        """Imputa los valores faltantes en las columnas numéricas utilizando KNNImputer."""
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            imputed_values = self.imputer.fit_transform(self.X[numeric_cols])
            self.X[numeric_cols] = imputed_values

    def standardize_numeric_data(self, ignore_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Estandariza las columnas numéricas excluyendo las columnas especificadas.
        Retorna el DataFrame estandarizado y guarda el modelo de estandarización.
        """
        if ignore_columns is None:
            ignore_columns = []
        numeric_cols = self.X.select_dtypes(include=[float, int]).columns.difference(ignore_columns)
        if not numeric_cols.empty:
            self.stored_scaler = StandardScaler()  # Crear una instancia nueva
            self.stored_scaler.fit(self.X[numeric_cols])
            self.X[numeric_cols] = self.stored_scaler.transform(self.X[numeric_cols])
        return self.X

    def normalize_numeric_data(self, ignore_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normaliza las columnas numéricas excluyendo las columnas especificadas.
        Retorna el DataFrame normalizado y guarda el modelo de normalización.
        """
        if ignore_columns is None:
            ignore_columns = []
        numeric_cols = self.X.select_dtypes(include=[float, int]).columns.difference(ignore_columns)
        if not numeric_cols.empty:
            self.stored_normalizer = MinMaxScaler()  # Crear una instancia nueva
            self.stored_normalizer.fit(self.X[numeric_cols])
            self.X[numeric_cols] = self.stored_normalizer.transform(self.X[numeric_cols])
        return self.X

    def reverse_standardize(self) -> pd.DataFrame:
        """
        Revierte la estandarización de las columnas numéricas.
        """
        if self.stored_scaler:
            numeric_cols = self.X.select_dtypes(include=[float, int]).columns
            self.X[numeric_cols] = self.stored_scaler.inverse_transform(self.X[numeric_cols])
        return self.X

    def reverse_normalize(self) -> pd.DataFrame:
        """
        Revierte la normalización de las columnas numéricas.
        """
        if self.stored_normalizer:
            numeric_cols = self.X.select_dtypes(include=[float, int]).columns
            self.X[numeric_cols] = self.stored_normalizer.inverse_transform(self.X[numeric_cols])
        return self.X

    def remove_duplicate_rows(self) -> None:
        """Elimina filas duplicadas en el DataFrame."""
        self.X.drop_duplicates(inplace=True)

    def remove_null_rows(self) -> None:
        """Elimina filas que contienen valores nulos en el DataFrame."""
        self.X.dropna(inplace=True)

    def apply_encoding(self, encoding_method: str = 'label', columns: Optional[List[str]] = None) -> (pd.DataFrame, Encoder):
        """
        Aplica el método de codificación (One-Hot, Label o Frequency) a las columnas especificadas.
        Retorna el DataFrame codificado y el encoder utilizado.
        """

        if columns is None:
            columns = self.X.select_dtypes(include=['object']).columns.tolist()
        if encoding_method == 'label':
            self.X = self.encoder.label_encode(self.X, columns)
        elif encoding_method == 'onehot':
            self.X = self.encoder.one_hot_encode(self.X, columns)
        elif encoding_method == 'frequency':
            self.X = self.encoder.frequency_encode(self.X, columns)
        return self.X, self.encoder  

    def reverse_encoding(self, encoding_method: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Decodifica las columnas codificadas de vuelta a sus valores originales.
        """
        if columns is None:
            columns = self.encoder.last_features_encoded
        if encoding_method == 'label':
            self.X = self.encoder.reverse_label_encode(self.X, columns)
        elif encoding_method == 'onehot':
            self.X = self.encoder.reverse_one_hot_encode(self.X, columns)
        elif encoding_method == 'frequency':
            self.X = self.encoder.reverse_frequency_encode(self.X, columns)
        return self.X



