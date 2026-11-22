import json

import numpy as np
import pandas as pd

class _TreeNode:
    def __init__(self, class_weights, n_samples, depth, impurity):
        self.class_weights = class_weights          
        self.weight = float(class_weights.sum())    
        self.n_samples = int(n_samples)             
        self.depth = depth
        self.impurity = float(impurity)
        self.feature = None                         
        self.threshold = None                       
        self.children = {}                          
        self.branch_weights = {}                    
        self.gain = 0.0                             

    @property
    def is_leaf(self):
        return self.feature is None

    @property
    def proba(self):
        return self.class_weights / self.weight

    def make_leaf(self):
        self.feature = None
        self.threshold = None
        self.children = {}
        self.branch_weights = {}
        self.gain = 0.0


class DecisionTreeClassifier:
    def __init__(self, criterion="gain_ratio", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain_ratio=1e-4, max_features=None,
                 feature_names=None, random_state=None, ccp_alpha=0.0,
                 class_weight=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.max_features = max_features
        self.feature_names = feature_names
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight

    _PARAM_NAMES = ("criterion", "max_depth", "min_samples_split", "min_samples_leaf",
                    "min_gain_ratio", "max_features", "feature_names", "random_state",
                    "ccp_alpha", "class_weight")

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self._PARAM_NAMES:
                raise ValueError(f"Parameter '{key}' tidak valid untuk DecisionTreeClassifier.")
            setattr(self, key, value)
        return self

    def _validate_params(self):
        if self.criterion not in ("gain_ratio", "entropy", "gini"):
            raise ValueError("criterion harus 'gain_ratio', 'entropy', atau 'gini'.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth harus >= 1 atau None.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split harus >= 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf harus >= 1.")
        if self.ccp_alpha is not None and self.ccp_alpha < 0:
            raise ValueError("ccp_alpha harus >= 0.")

    def _check_fitted(self):
        if not hasattr(self, "root_"):
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")
    
    def _check_fitted(self):
        if not hasattr(self, "root_"):
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")

    def _frame(self, X, fitting):
        """Ubah input ke DataFrame; pada prediksi, urutkan kolom sesuai saat fit."""
        if isinstance(X, pd.DataFrame):
            df = X
            if not fitting:
                names = list(self.feature_names_)
                cols = [str(c) for c in df.columns]
                if set(names).issubset(cols):
                    df = df.iloc[:, [cols.index(n) for n in names]]
                elif df.shape[1] != self.n_features_in_:
                    raise ValueError(f"Jumlah fitur ({df.shape[1]}) != saat fit ({self.n_features_in_}).")
            return df
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X harus 2 dimensi.")
        if not fitting and arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Jumlah fitur ({arr.shape[1]}) != saat fit ({self.n_features_in_}).")
        df = pd.DataFrame(arr)
        if fitting and arr.dtype == object:
            for j in df.columns:
                try:
                    df[j] = pd.to_numeric(df[j])
                except (ValueError, TypeError):
                    pass
        return df

    def _setup_features(self, df):
        self.n_features_in_ = df.shape[1]
        if isinstance(df.columns, pd.RangeIndex):
            default = [f"x{i}" for i in range(df.shape[1])]
            names = list(self.feature_names) if self.feature_names is not None else default
        else:
            names = [str(c) for c in df.columns]
        if len(names) != self.n_features_in_:
            raise ValueError("feature_names tidak sesuai jumlah kolom X.")
        self.feature_names_ = names

        self.is_numeric_feature_, self.categories_, self._cat_maps = [], [], []
        for j in range(self.n_features_in_):
            s = df.iloc[:, j]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                self.is_numeric_feature_.append(True)
                self.categories_.append(None)
                self._cat_maps.append(None)
            else:
                values = list(pd.unique(s.dropna()))
                try:
                    values = sorted(values)
                except TypeError:
                    pass
                self.is_numeric_feature_.append(False)
                self.categories_.append(values)
                self._cat_maps.append({v: i for i, v in enumerate(values)})

    def _encode_column(self, s, j):
        if self.is_numeric_feature_[j]:
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        mapped = pd.to_numeric(s.astype(object).map(self._cat_maps[j]), errors="coerce")
        codes = mapped.to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isnan(codes), -1, codes).astype(np.int64)

    def _sample_weights(self, sample_weight, y_idx):
        n = len(y_idx)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float).copy()
        if len(w) != n:
            raise ValueError("sample_weight harus sepanjang y.")
        if (w < 0).any():
            raise ValueError("sample_weight tidak boleh negatif.")
        if self.class_weight is not None:
            counts = np.bincount(y_idx, minlength=self._n_classes)
            if isinstance(self.class_weight, str):
                if self.class_weight != "balanced":
                    raise ValueError("class_weight string hanya mendukung 'balanced'.")
                cw = n / (self._n_classes * np.maximum(counts, 1))
            else:
                cw = np.array([self.class_weight.get(c, self.class_weight.get(c.item() if hasattr(c, "item") else c, 1.0))
                               for c in self.classes_], dtype=float)
            w = w * cw[y_idx]
        return w

    def _resolve_max_features(self):
        n, mf = self.n_features_in_, self.max_features
        if mf is None:
            return n
        if mf == "sqrt":
            return max(1, int(np.sqrt(n)))
        if mf == "log2":
            return max(1, int(np.log2(n)))
        if isinstance(mf, (int, np.integer)):
            return int(min(max(mf, 1), n))
        if isinstance(mf, float):
            return int(min(max(1, int(mf * n)), n))
        raise ValueError("max_features tidak valid.")