import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Clase OutlierDetector para manejar outliers y rastrear qué columnas causan la eliminación
class OutlierDetector:
    """
    Clase para manejar la detección y eliminación de outliers utilizando diferentes métodos.
    También permite rastrear qué columnas específicas causaron la eliminación de cada fila.
    """

    @staticmethod
    def detect_and_remove_outliers(X: pd.DataFrame, method: str = 'zscore', ignore_columns: list = None, **kwargs) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers utilizando el método especificado, ignorando las columnas indicadas en ignore_columns.
        Retorna un DataFrame con las filas que fueron eliminadas como outliers y la columna que causó la eliminación.
        
        Args:
            X (pd.DataFrame): El DataFrame con los datos a analizar.
            method (str): El método de detección de outliers a utilizar ('zscore', 'iqr', 'isolation_forest').
            ignore_columns (list): Lista de columnas que no serán consideradas para la detección de outliers.
            kwargs: Parámetros adicionales para los métodos de outlier detection.
        
        Returns:
            pd.DataFrame: Un DataFrame que contiene las filas eliminadas.
            pd.DataFrame: Un DataFrame que contiene la columna que causó la eliminación del outlier.
        """
        # Si no se especifican columnas a ignorar, se establece una lista vacía
        if ignore_columns is None:
            ignore_columns = []

        # Eliminar las columnas a ignorar del conjunto de datos
        X_to_analyze = X.drop(columns=ignore_columns, errors='ignore')

        # Aplicar el método de eliminación de outliers
        if method == 'zscore':
            return OutlierDetector._detect_outliers_zscore(X, X_to_analyze, **kwargs)
        elif method == 'iqr':
            return OutlierDetector._detect_outliers_iqr(X, X_to_analyze, **kwargs)
        elif method == 'isolation_forest':
            return OutlierDetector._detect_outliers_isolation_forest(X, X_to_analyze, **kwargs)

    @staticmethod
    def _detect_outliers_zscore(X: pd.DataFrame, X_to_analyze: pd.DataFrame, threshold: float = 3.0) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers usando Z-score y rastrea qué columna causó la eliminación.
        
        Args:
            X (pd.DataFrame): El DataFrame original.
            X_to_analyze (pd.DataFrame): El DataFrame sobre el que se realiza la detección de outliers.
            threshold (float): El umbral para considerar un valor como outlier basado en el Z-score. Default es 3.0.
        
        Returns:
            pd.DataFrame: Las filas eliminadas.
            pd.DataFrame: Las razones (columnas) que causaron la eliminación.
        """
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0))
        
        # Filas que tienen al menos un valor de Z-score mayor al umbral
        outliers = (z_scores > threshold).any(axis=1)
        
        # Guardar los outliers eliminados
        removed_outliers = X[outliers]
        
        # Identificar las columnas responsables de la eliminación por outliers
        removal_reasons = z_scores[outliers].idxmax(axis=1)
        removal_reasons_df = pd.DataFrame({'Column': removal_reasons})

        # Eliminar los outliers del DataFrame original
        X_cleaned = X[~outliers]
        
        return removed_outliers, removal_reasons_df

    @staticmethod
    def _detect_outliers_iqr(X: pd.DataFrame, X_to_analyze: pd.DataFrame, multiplier: float = 1.5) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers usando el rango intercuartílico (IQR) y rastrea qué columna causó la eliminación.
        
        Args:
            X (pd.DataFrame): El DataFrame original.
            X_to_analyze (pd.DataFrame): El DataFrame sobre el que se realiza la detección de outliers.
            multiplier (float): El multiplicador de IQR para identificar los outliers. Default es 1.5.
        
        Returns:
            pd.DataFrame: Las filas eliminadas.
            pd.DataFrame: Las razones (columnas) que causaron la eliminación.
        """
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = ((numeric_cols < (Q1 - multiplier * IQR)) | (numeric_cols > (Q3 + multiplier * IQR)))
        
        # Filas que tienen al menos un valor fuera del rango IQR
        outlier_rows = is_outlier.any(axis=1)
        
        # Guardar los outliers eliminados
        removed_outliers = X[outlier_rows]
        
        # Identificar las columnas responsables de la eliminación por outliers
        removal_reasons = is_outlier[outlier_rows].idxmax(axis=1)
        removal_reasons_df = pd.DataFrame({'Column': removal_reasons})

        # Eliminar los outliers del DataFrame original
        X_cleaned = X[~outlier_rows]
        
        return removed_outliers, removal_reasons_df

    @staticmethod
    def _detect_outliers_isolation_forest(X: pd.DataFrame, X_to_analyze: pd.DataFrame, contamination: float = 0.01) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers usando Isolation Forest. Este método no rastrea específicamente la columna
        que causó la eliminación, ya que es un algoritmo basado en todo el conjunto de características.
        
        Args:
            X (pd.DataFrame): El DataFrame original.
            X_to_analyze (pd.DataFrame): El DataFrame sobre el que se realiza la detección de outliers.
            contamination (float): La proporción de los datos que se espera que sean outliers. Default es 0.01.
        
        Returns:
            pd.DataFrame: Las filas eliminadas.
            pd.DataFrame: Las razones (en este caso, siempre será 'Multiple').
        """
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(numeric_cols)
        
        # Guardar los outliers eliminados
        removed_outliers = X[outlier_pred == -1]
        
        # Para Isolation Forest no se puede identificar una columna específica responsable
        removal_reasons_df = pd.DataFrame({'Column': ['Multiple'] * len(removed_outliers)})

        # Eliminar los outliers del DataFrame original
        X_cleaned = X[outlier_pred != -1]
        
        return removed_outliers, removal_reasons_df

    @staticmethod
    def print_removed_outliers_with_reasons(removed_outliers: pd.DataFrame, removal_reasons: pd.DataFrame) -> None:
        """
        Imprime las filas eliminadas junto con la columna que causó su eliminación como outlier.
        
        Args:
            removed_outliers (pd.DataFrame): Las filas eliminadas por outliers.
            removal_reasons (pd.DataFrame): Las razones por las que se eliminaron los outliers.
        """
        if not removed_outliers.empty:
            print("\nOutliers eliminados:")
            print(removed_outliers)
            print("\nRazón (columna que causó la eliminación):")
            print(removal_reasons)
        else:
            print("No se eliminaron outliers.")

    @staticmethod
    def show_outliers_by_column(X: pd.DataFrame, removed_outliers: pd.DataFrame, removal_reasons: pd.DataFrame, column_name: str) -> None:
        """
        Muestra los outliers eliminados que están relacionados con una columna específica.
        También imprime la media de la columna antes de la eliminación de outliers.
        
        Args:
            X (pd.DataFrame): El DataFrame original.
            removed_outliers (pd.DataFrame): Las filas eliminadas por outliers.
            removal_reasons (pd.DataFrame): Las razones por las que se eliminaron los outliers.
            column_name (str): El nombre de la columna en la que se desea enfocar la detección de outliers.
        """
        # Verificar si la columna existe
        if column_name not in X.columns:
            print(f"La columna '{column_name}' no existe en el DataFrame.")
            return

        # Mostrar la media de la columna antes de la eliminación de outliers
        print(f"\nMedia de '{column_name}' antes de eliminar outliers: {X[column_name].mean()}")

        # Buscar filas eliminadas donde la columna fue la razón del outlier
        if not removal_reasons.empty:
            related_outliers = removal_reasons[removal_reasons['Column'] == column_name]
            
            if not related_outliers.empty:
                # Mostrar los outliers relacionados a esa columna
                related_outlier_rows = removed_outliers.loc[related_outliers.index]
                print(f"\nOutliers eliminados para la columna '{column_name}':")
                print(related_outlier_rows[column_name])
            else:
                print(f"No se encontraron outliers eliminados relacionados con la columna '{column_name}'.")
        else:
            print("No se han eliminado outliers o no se ha registrado información sobre las columnas.")
