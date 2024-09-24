import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Clase OutlierDetector para manejar outliers y rastrear qué columnas causan la eliminación
class OutlierDetector:
    """
    Clase para manejar la detección y eliminación de outliers utilizando diferentes métodos.
    Permite rastrear qué columnas específicas causaron la eliminación de cada fila.
    """
    def __init__(self):
        self.removed_outliers = pd.DataFrame()
        self.removal_reasons = pd.DataFrame()

    def detect_and_remove_outliers(self, X: pd.DataFrame, method: str = 'zscore', ignore_columns: list = None, **kwargs) -> pd.DataFrame:
        """
        Detecta y elimina outliers utilizando el método especificado, ignorando las columnas indicadas.
        Guarda los datos eliminados y la columna que causó la eliminación.

        Args:
            X (pd.DataFrame): El DataFrame con los datos a analizar.
            method (str): El método de detección de outliers a utilizar ('zscore', 'iqr', 'isolation_forest').
            ignore_columns (list): Lista de columnas que no serán consideradas para la detección de outliers.
            kwargs: Parámetros adicionales para los métodos de detección.

        Returns:
            pd.DataFrame: Un DataFrame limpio sin los outliers.
        """
        if ignore_columns is None:
            ignore_columns = []

        X_to_analyze = X.drop(columns=ignore_columns, errors='ignore')

        if method == 'zscore':
            self.removed_outliers, self.removal_reasons = self._detect_outliers_zscore(X, X_to_analyze, **kwargs)
        elif method == 'iqr':
            self.removed_outliers, self.removal_reasons = self._detect_outliers_iqr(X, X_to_analyze, **kwargs)
        elif method == 'isolation_forest':
            self.removed_outliers, self.removal_reasons = self._detect_outliers_isolation_forest(X, X_to_analyze, **kwargs)

        X_cleaned = X.drop(self.removed_outliers.index)
        return X_cleaned

    def _detect_outliers_zscore(self, X: pd.DataFrame, X_to_analyze: pd.DataFrame, threshold: float = 3.0) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers usando Z-score y rastrea qué columna causó la eliminación.
        """
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0))
        outliers = (z_scores > threshold).any(axis=1)
        removed_outliers = X[outliers]
        removal_reasons = z_scores[outliers].idxmax(axis=1)
        removal_reasons_df = pd.DataFrame({'Column': removal_reasons})
        return removed_outliers, removal_reasons_df

    def _detect_outliers_iqr(self, X: pd.DataFrame, X_to_analyze: pd.DataFrame, multiplier: float = 1.5) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers usando el rango intercuartílico (IQR) y rastrea qué columna causó la eliminación.
        """
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = ((numeric_cols < (Q1 - multiplier * IQR)) | (numeric_cols > (Q3 + multiplier * IQR)))
        outlier_rows = is_outlier.any(axis=1)
        removed_outliers = X[outlier_rows]
        removal_reasons = is_outlier[outlier_rows].idxmax(axis=1)
        removal_reasons_df = pd.DataFrame({'Column': removal_reasons})
        return removed_outliers, removal_reasons_df

    def _detect_outliers_isolation_forest(self, X: pd.DataFrame, X_to_analyze: pd.DataFrame, contamination: float = 0.01) -> (pd.DataFrame, pd.DataFrame):
        """
        Detecta y elimina outliers usando Isolation Forest. No rastrea específicamente la columna que causó la eliminación.
        """
        numeric_cols = X_to_analyze.select_dtypes(include=[np.number])
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(numeric_cols)
        removed_outliers = X[outlier_pred == -1]
        removal_reasons_df = pd.DataFrame({'Column': ['Multiple'] * len(removed_outliers)})
        return removed_outliers, removal_reasons_df

    def show_outliers_by_column(self, column_name: str) -> None:
        """
        Muestra los outliers eliminados que están relacionados con una columna específica.
        """
        if column_name not in self.removal_reasons['Column'].values:
            print(f"No se encontraron outliers eliminados relacionados con la columna '{column_name}'.")
            return

        related_outliers = self.removal_reasons[self.removal_reasons['Column'] == column_name]
        related_outlier_rows = self.removed_outliers.loc[related_outliers.index]
        print(f"\nOutliers eliminados para la columna '{column_name}':")
        print(related_outlier_rows[column_name])

# Clase para la visualización de outliers
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierVisualizer:
    """
    Clase para visualizar los outliers eliminados en gráficos de dispersión, coloreados por la columna que causó la eliminación.
    """

    @staticmethod
    def plot_subset_outliers(removed_outliers: pd.DataFrame, removal_reasons: pd.DataFrame, num_features: int = 2) -> None:
        """
        Crea gráficos de dispersión seleccionando de forma aleatoria un número de características a visualizar.

        Args:
            removed_outliers (pd.DataFrame): Las filas eliminadas por outliers.
            removal_reasons (pd.DataFrame): Las razones (columnas) que causaron la eliminación.
            num_features (int): Número de características a seleccionar para la visualización.
        """
        if removed_outliers.empty:
            print("No se encontraron outliers eliminados.")
            return

        # Seleccionar un subconjunto de características numéricas para visualizar
        available_features = removed_outliers.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(available_features) < num_features:
            num_features = len(available_features)  # Ajusta si hay menos características que las solicitadas
        
        # Seleccionar aleatoriamente características para graficar
        selected_features = random.sample(available_features, num_features)

        # Crear gráficos de dispersión para el subconjunto de características seleccionadas
        for i in range(len(selected_features) - 1):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=removed_outliers, 
                            x=selected_features[i], 
                            y=selected_features[i + 1], 
                            hue=removal_reasons['Column'], 
                            palette='Set1')
            plt.title(f'Outliers eliminados en {selected_features[i]} vs {selected_features[i + 1]}')
            plt.xlabel(selected_features[i])
            plt.ylabel(selected_features[i + 1])
            plt.legend(title="Razón de eliminación")
            plt.show()