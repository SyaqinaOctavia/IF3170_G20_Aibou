
import json
import math
import pandas as pd
import numpy as np

import graphviz

from pathlib import Path


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9, alpha=1.0, priors: dict[str, float] | None = None):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.priors = priors

    def fit(self, X, y):
        if len(X) != len(X):
            raise ValueError("X and y are not the same length")

        total = len(y)

        self._target_name = y.name
        self._classes = np.asarray(y.unique()).tolist()
        self._class_prior = {}
        self._feature_types = {}

        self._mean = {}
        self._variance = {}
        self._cat_prob = {}

        for col in np.asarray(X.columns):
            self._feature_types[col] = 'numeric' if pd.api.types.is_numeric_dtype(X[col]) else 'categorical'
        for c in self._classes:
            c_key = str(c)
            self._class_prior[c_key] = len(y[y == c]) / total if total > 0 else 0.0

        fit_df = X.join(y)
        target_col = y.name
        for c in self._classes:
            c_key = str(c)
            c_df = fit_df[fit_df[target_col] == c]

            N_c = len(c_df)
            self._mean[c_key] = {}
            self._variance[c_key] = {}
            self._cat_prob[c_key] = {}
            for col, ftype in self._feature_types.items():
                col_df = c_df[col]
                if ftype == 'numeric':
                    self._mean[c_key][col] = col_df.mean()
                    self._variance[c_key][col] = col_df.var()
                elif ftype == 'categorical':
                    self._cat_prob[c_key][col] = {}
                    ni = len(col_df.unique())
                    for value in col_df.unique():
                        key = str(value)
                        if not pd.isna(value):
                            N_itc = len(col_df[col_df == value])
                        else :
                            key = "__nan__"
                            N_itc = len(col_df[pd.isna(col_df)]) 
                        self._cat_prob[c_key][col][key] = (N_itc + self.alpha) / (N_c + self.alpha * ni)  
                else:
                    assert False
        
                
    def predict(self, X, use_log_space=False):
        for col, ftype in self._feature_types.items():
            if col not in np.asarray(X.columns):
                raise IndexError("X does not have column named " + col)
            X_ftype = 'numeric' if pd.api.types.is_numeric_dtype(X[col]) else 'categorical'
            if ftype != X_ftype:
                raise TypeError(f'Column {col} of X has type of {X_ftype} instead of {ftype}')

        result = []
        for _, row in X.iterrows():
            result.append(self._predict_proba(row, use_log_space))
        return result

    def _predict_proba(self, X_row, use_log_space=False):
        p_best = 0
        c_best = self._classes[0]
        for c in self._classes:
            c_key = str(c)
            if self.priors is not None:
                p_current = self.priors.get(c, self.priors.get(c_key, self._class_prior[c_key]))
            else:
                p_current = self._class_prior[c_key]
            if use_log_space:
                p_current = math.log(p_current)
            for col, ftype in self._feature_types.items():
                value = X_row[col]
                if ftype == 'numeric':
                    mean = self._mean[c_key][col]
                    var = self._variance[c_key][col] + self.var_smoothing
                    if pd.isna(value):
                        value = mean
                    if not use_log_space:
                        p_current *= (
                            1 / math.sqrt(2 * math.pi * var) 
                            * math.exp(-(value - mean)**2 / (2 * var))
                        )
                    else:
                        p_current += -1/2 * math.log(2 * math.pi * var) - (value - mean)**2 / (2 * var)
                elif ftype == 'categorical':
                    key = value if not pd.isna(value) else "__nan__"
                    if not use_log_space:
                        p_current *= self._cat_prob[c_key][col].get(key, 0)
                    else:
                        p_current += math.log(self._cat_prob[c_key][col].get(key, 0) + self.var_smoothing)
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
                'mean': self._mean,
                'variance': self._variance,
                'cat_prob': self._cat_prob,
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

        model._mean = state.get('mean')
        model._variance = state.get('variance')
        model._cat_prob = state.get('cat_prob')

        return model

    def display(self):
        graph = graphviz.Digraph(graph_attr={'rankdir': 'LR'})

        for c in self._classes:
            c_key = str(c)
            if self.priors is not None:
                p_current = self.priors.get(c, self.priors.get(c_key, self._class_prior[c_key]))
            else:
                p_current = self._class_prior[c_key]
            graph.node(f'{self._target_name}={c}', f'{self._target_name}={c}\nP={p_current}')
            for col, ftype in self._feature_types.items():
                if ftype == 'numeric':
                    mean = self._mean[c_key][col]
                    var = self._variance[c_key][col]
                    graph.node(f'{self._target_name}={c},{col}=x', f'{col}=x')
                    graph.edge(f'{self._target_name}={c}', f'{self._target_name}={c},{col}=x', label=f'μ={mean}, σ²={var}')
                elif ftype == 'categorical':
                    for value in self._cat_prob[c_key][col].keys():
                        key = value if value != '__nan__' else 'nan'
                        graph.node(f'{self._target_name}={c},{col}={key}', f'{col}={key}')
                        graph.edge(f'{self._target_name}={c}', f'{self._target_name}={c},{col}=x', label=f'P={self._cat_prob[c_key][col].get(value, 0)}')
                else:
                    assert False
        return graph
