from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Union, List
import numpy as np
import pandas as pd

# ============================================================================
# Data Preprocessing (self-contained, no external dependencies)
# ============================================================================

class DataPreprocessor:
    """
    Preprocessor for tabular data before clustering.
    
    Handles:
    - Categorical encoding (label encoding)
    - NaN imputation
    - Zero-variance column removal
    - Normalization (z-norm by default)
    """
    
    def __init__(
        self,
        categorical_encoding: str = "label",
        nan_strategy: str = "zero",
        normalization: Union[str, List[str]] = "z-norm",
        remove_zero_variance: bool = True,
    ):
        self.categorical_encoding = categorical_encoding
        self.nan_strategy = nan_strategy
        if isinstance(normalization, str):
            self.normalization = [normalization] if normalization != "none" else []
        else:
            self.normalization = normalization
        self.remove_zero_variance = remove_zero_variance

    def _split_columns(self, X: pd.DataFrame):
        """Split columns into categorical and numeric."""
        categorical_cols = []
        numeric_cols = []
        for col in X.columns:
            dtype = X[col].dtype
            if dtype.name == "category" or dtype == "object" or dtype.name == "string":
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        return categorical_cols, numeric_cols

    def _encode_categorical(self, X: pd.DataFrame, categorical_cols, numeric_cols) -> tuple:
        """Encode categorical features using label encoding."""
        if not categorical_cols:
            X_array = X[numeric_cols].to_numpy(dtype=np.float64) if numeric_cols else np.empty((len(X), 0))
            return X_array, list(range(X_array.shape[1]))

        numeric_indices = []
        col_idx = 0
        parts = []
        
        for col in X.columns:
            if col in categorical_cols:
                parts.append(LabelEncoder().fit_transform(X[col].astype(str)).astype(np.float64))
            else:
                parts.append(X[col].to_numpy(dtype=np.float64))
            numeric_indices.append(col_idx)
            col_idx += 1
            
        return np.column_stack(parts) if parts else np.empty((len(X), 0)), numeric_indices

    def _handle_nan(self, X: pd.DataFrame, categorical_cols, numeric_cols) -> pd.DataFrame:
        """Handle NaN values in the data."""
        X_filled = X.copy()
        
        # Numeric columns
        if numeric_cols:
            if self.nan_strategy == "zero":
                X_filled[numeric_cols] = X_filled[numeric_cols].fillna(0.0)
            elif self.nan_strategy in {"mean", "median", "mode"}:
                strategy = {"mean": "mean", "median": "median", "mode": "most_frequent"}[self.nan_strategy]
                imputer = SimpleImputer(strategy=strategy)
                X_filled[numeric_cols] = imputer.fit_transform(X_filled[numeric_cols])
                
        # Categorical columns: use mode
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            imputed = cat_imputer.fit_transform(X_filled[categorical_cols])
            imputed_df = pd.DataFrame(imputed, columns=categorical_cols, index=X_filled.index)
            for col in categorical_cols:
                X_filled[col] = imputed_df[col].astype("category")
                
        return X_filled

    def _remove_zero_variance(self, X: np.ndarray, numeric_feature_indices: List[int]) -> tuple:
        """Remove zero-variance columns."""
        if not self.remove_zero_variance:
            return X, numeric_feature_indices
        
        variances = np.var(X, axis=0)
        mask = variances > 1e-9
        if not np.any(mask):
            raise ValueError("All columns have zero variance, no features left")
        
        X_filtered = X[:, mask]
        kept_indices = np.where(mask)[0]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
        updated_numeric_indices = [old_to_new[idx] for idx in numeric_feature_indices if idx in old_to_new]
        
        return X_filtered, updated_numeric_indices

    def _normalize(self, X: np.ndarray, numeric_feature_indices: List[int]) -> np.ndarray:
        """Apply normalization to numeric features."""
        if not self.normalization:
            return X
        
        if numeric_feature_indices is None:
            numeric_feature_indices = list(range(X.shape[1]))
        
        X_normalized = X.copy()
        
        for norm_step in self.normalization:
            if norm_step == "unit_variance":
                X_centered = X_normalized - np.mean(X_normalized, axis=0, keepdims=True)
                total_var = np.var(X_centered)
                if total_var > 1e-8:
                    X_normalized = X_centered / np.sqrt(total_var)
                else:
                    X_normalized = X_centered
            elif norm_step == "z-norm":
                if numeric_feature_indices:
                    scaler = StandardScaler()
                    X_normalized[:, numeric_feature_indices] = scaler.fit_transform(
                        X_normalized[:, numeric_feature_indices]
                    )
            elif norm_step == "minmax":
                if numeric_feature_indices:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    X_normalized[:, numeric_feature_indices] = scaler.fit_transform(
                        X_normalized[:, numeric_feature_indices]
                    )
        
        return X_normalized

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess the data for clustering."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        categorical_cols, numeric_cols = self._split_columns(X)
        X_no_nan = self._handle_nan(X, categorical_cols, numeric_cols)
        X_encoded, numeric_feature_indices = self._encode_categorical(X_no_nan, categorical_cols, numeric_cols)
        X_no_zero_var, numeric_feature_indices = self._remove_zero_variance(X_encoded, numeric_feature_indices)
        X_normalized = self._normalize(X_no_zero_var, numeric_feature_indices)
        return X_normalized
