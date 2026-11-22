
import json
import math
import pandas as pd
import numpy as np

from pathlib import Path


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9, alpha=1.0, priors=None):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.priors = priors

        self._classes = []
        self._class_prior = []

        self._feature_types = dict[str, str]()
        self._mean = {}
        self._variance = {}
        self._cat_prob = {}

    def fit(self, X, y):
        if len(X) != len(X):
            print("X and y are not the same length")
            return

        total = len(X)

        self._feature_types = {}
        for col in np.asarray(X.columns):
            self._feature_types[col] = 'numeric' if pd.api.types.is_numeric_dtype(X[col]) else 'categorical'
        self._classes = np.asarray(y.unique())

        self._p0 = {}
        self._count = {}
        for c in self._classes:
            self._count[c] = len(y[y == c])
            self._p0[c] = self._count[c] / total if total > 0 else 0.0
        
        self._mean = {}
        self._variance = {}
        self._cat_prob = {}
        self._cat_count = {}

        fit_df = X.join(y)
        target_col = y.name
        for c in self._classes:
            c_df = fit_df[fit_df[target_col] == c]
            self._mean[c] = {}
            self._variance[c] = {}
            self._cat_prob[c] = {}
            self._cat_count[c] = {}
            for col, ftype in self._feature_types.items():
                col_df = c_df[col]
                if ftype == 'numeric':
                    self._mean[c][col] = col_df.mean()
                    self._variance[c][col] = col_df.var()
                elif ftype == 'categorical':
                    self._cat_prob[c][col] = {}
                    for value in col_df.unique():
                        self._cat_count[c][value] = len(col_df[col_df == value])
                        self._cat_prob[c][col][value] = self._cat_count[c][value] / total if total > 0 else 0.0
                else:
                    assert False
        
                
    def predict(self, X):
        result = []
        for _, row in X.iterrows():
            result.append(self._predict_row(row))
        return result

    def _predict_row(self, X_row):
        p_best = 0
        c_best = self._classes[0]
        for c in self._classes:
            p_current = self._p0[c]
            for col, ftype in self._feature_types.items():
                value = X_row[col]
                if ftype == 'numeric':
                    mean = self._mean[c][col]
                    var = self._variance[c][col] + self.var_smoothing
                    p_current = (
                        p_current 
                        * 1 / math.sqrt(2 * math.pi * var) 
                        * math.exp(- (value - mean)**2 / (2 * var))
                    )
                elif ftype == 'categorical':
                    p_current = (
                        p_current 
                        * (self._cat_prob[c][col].get(value, 0) + self.var_smoothing)
                        / (self._count[c] + self.var_smoothing * self._cat_count[c].get(value, 0))  
                    )
                else:
                    assert False
                
            if p_current > p_best:
                p_best = p_current
                c_best = c
        return c_best