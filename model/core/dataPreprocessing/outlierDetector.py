import pandas as pd
import numpy as np

# Clase OutlierDetector para manejar outliers
class OutlierDetector:
    """
    Clase para manejar la detección y eliminación de outliers utilizando diferentes métodos.
    """
    def __init__(self, X: pd.DataFrame):
        self.X = X
        self.removed_outliers = pd.DataFrame()  # Para almacenar las filas eliminadas

    def detect_and_remove_outliers(self, method: str = 'isolation_forest', **kwargs) -> pd.DataFrame:
        """
        Detecta y elimina outliers utilizando el método especificado.
        Retorna las filas eliminadas.
        """
        if method == 'zscore':
            self._detect_outliers_zscore(**kwargs)
        elif method == 'iqr':
            self._detect_outliers_iqr(**kwargs)
        elif method == 'isolation_forest':
            self._detect_outliers_isolation_forest(**kwargs)
        
        return self.removed_outliers

    def _detect_outliers_zscore(self, threshold: float = 3.0) -> None:
        """Detecta y elimina outliers usando Z-score."""
        numeric_cols = self.X.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0))
        outliers = (z_scores > threshold).any(axis=1)
        
        # Guardar los outliers eliminados
        self.removed_outliers = self.X[outliers]
        
        # Eliminar los outliers del DataFrame original
        self.X = self.X[~outliers]

    def _detect_outliers_iqr(self, multiplier: float = 1.5) -> None:
        """Detecta y elimina outliers usando el rango intercuartílico (IQR)."""
        numeric_cols = self.X.select_dtypes(include=[np.number])
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = ((numeric_cols < (Q1 - multiplier * IQR)) | (numeric_cols > (Q3 + multiplier * IQR))).any(axis=1)
        
        # Guardar los outliers eliminados
        self.removed_outliers = self.X[is_outlier]
        
        # Eliminar los outliers del DataFrame original
        self.X = self.X[~is_outlier]

    def _detect_outliers_isolation_forest(self, contamination: float = 0.01) -> None:
        """Detecta y elimina outliers usando Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        numeric_cols = self.X.select_dtypes(include=[np.number])
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(numeric_cols)
        
        # Guardar los outliers eliminados
        self.removed_outliers = self.X[outlier_pred == -1]
        
        # Eliminar los outliers del DataFrame original
        self.X = self.X[outlier_pred != -1]