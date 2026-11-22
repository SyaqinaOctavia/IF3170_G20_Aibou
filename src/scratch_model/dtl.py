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