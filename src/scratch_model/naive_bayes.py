
import json
import math
import pandas as pd
import numpy as np

from pathlib import Path


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9, alpha=1.0, priors: dict[str, float] | None = None):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.priors = priors

    def fit(self, X, y):
        if len(X) != len(X):
            raise ValueError("X and y are not the same length")

        total = len(X)

        self._classes = np.asarray(y.unique()).tolist()
        self._class_prior = {}
        self._class_count = {}
        self._feature_types = {}

        self._mean = {}
        self._variance = {}
        self._cat_prob = {}
        self._cat_count = {}

        for col in np.asarray(X.columns):
            self._feature_types[col] = 'numeric' if pd.api.types.is_numeric_dtype(X[col]) else 'categorical'
        for c in self._classes:
            c_key = str(c)
            self._class_count[c_key] = len(y[y == c])
            self._class_prior[c_key] = self._class_count[c_key] / total if total > 0 else 0.0

        fit_df = X.join(y)
        target_col = y.name
        for c in self._classes:
            c_key = str(c)
            c_df = fit_df[fit_df[target_col] == c]
            self._mean[c_key] = {}
            self._variance[c_key] = {}
            self._cat_prob[c_key] = {}
            self._cat_count[c_key] = {}
            for col, ftype in self._feature_types.items():
                col_df = c_df[col]
                if ftype == 'numeric':
                    self._mean[c_key][col] = col_df.mean()
                    self._variance[c_key][col] = col_df.var()
                elif ftype == 'categorical':
                    self._cat_count[c_key][col] = {}
                    self._cat_prob[c_key][col] = {}
                    for value in col_df.unique():
                        if not pd.isna(value):
                            self._cat_count[c_key][col][value] = len(col_df[col_df == value])
                            self._cat_prob[c_key][col][value] = self._cat_count[c_key][col][value] / total if total > 0 else 0.0
                        else :
                            key = "__nan__"
                            self._cat_count[c_key][col][key] = len(col_df[pd.isna(col_df)])
                            self._cat_prob[c_key][col][key] = self._cat_count[c_key][col][key] / total if total > 0 else 0.0
                else:
                    assert False
        
                
    def predict(self, X):
        for col, ftype in self._feature_types.items():
            if col not in np.asarray(X.columns):
                raise IndexError("X does not have column named " + col)
            X_ftype = 'numeric' if pd.api.types.is_numeric_dtype(X[col]) else 'categorical'
            if ftype != X_ftype:
                raise TypeError(f'Column {col} of X has type of {X_ftype} instead of {ftype}')

        result = []
        for _, row in X.iterrows():
            result.append(self._predict_row(row))
        return result

    def _predict_row(self, X_row):
        p_best = 0
        c_best = self._classes[0]
        for c in self._classes:
            c_key = str(c)
            if self.priors is not None:
                p_current = self.priors.get(c, self.priors.get(c_key, self._class_prior[c_key]))
            else:
                p_current = self._class_prior[c_key]
            for col, ftype in self._feature_types.items():
                value = X_row[col]
                if ftype == 'numeric':
                    mean = self._mean[c_key][col]
                    var = self._variance[c_key][col] + self.var_smoothing
                    if pd.isna(value):
                        value = mean
                    p_current = (
                        p_current 
                        * 1 / math.sqrt(2 * math.pi * var) 
                        * math.exp(-(value - mean)**2 / (2 * var))
                    )
                elif ftype == 'categorical':
                    key = value if not pd.isna(value) else "__nan__"
                    p_current = (
                        p_current 
                        * (self._cat_prob[c_key][col].get(key, 0) + self.alpha)
                        / (self._class_count[c_key] + self.alpha * self._cat_count[c_key][col].get(key, 0))  
                    )
                else:
                    assert False
                
            if p_current > p_best:
                p_best = p_current
                c_best = c
        return c_best

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_class': type(self).__name__,
            'state': {
                'var_smoothing': self.var_smoothing,
                'alpha': self.alpha,
                'priors': self.priors,
                'feature_types': self._feature_types,
                'classes': self._classes,
                'class_prior': self._class_prior,
                'class_count': self._class_count,
                'mean': self._mean,
                'variance': self._variance,
                'cat_prob': self._cat_prob,
                'cat_count': self._cat_count,
            }
        }
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)

    @staticmethod
    def load(path):
        target = Path(path)
        payload = json.loads(target.read_text(encoding='utf-8'))
        model_class = payload.get('model_class')
        if (model_class != 'NaiveBayes'):
            return None
        
        state = payload.get('state')

        model = NaiveBayes()
        model.var_smoothing = state.get('var_smoothing')
        model.alpha = state.get('alpha')
        model.priors = state.get('priors')
        
        model._feature_types = state.get('feature_types')
        model._classes = state.get('classes')

        model._class_prior = state.get('class_prior')
        model._class_count = state.get('class_count')

        model._mean = state.get('mean')
        model._variance = state.get('variance')
        model._cat_prob = state.get('cat_prob')
        model._cat_count = state.get('cat_count')

        return model

