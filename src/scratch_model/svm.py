import json
import pandas as pd
import numpy as np

from ._paths import resolve_model_path

class SVM:
    def __init__(self, C: float = 1.0, kernel: str = "rbf", degree: int = 3, gamma="scale", 
                 coef0: float = 0.0, max_iter: int = 500, learning_rate: float = 1e-3, 
                 tol: float = 1e-4, random_state: int | None = None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state

    def _gamma_value(self, X):
        if self.gamma == "scale":
            v = X.var()
            return 1.0 / (X.shape[1] * v) if v > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _kernel(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (self._g * (A @ B.T) + self.coef0) ** self.degree
        if self.kernel == "rbf":
            sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return np.exp(-self._g * np.maximum(sq, 0))
        raise ValueError("kernel must be 'linear', 'poly' or 'rbf'")

    # projection onto 0 <= a <= C, sum(a*y) = 0
    def _project(self, v, y):
        L = np.abs(v).max() + self.C + 1.0
        lo, hi = -L, L
        for _ in range(100):  # bisection on the multiplier of sum(a*y) = 0
            lam = (lo + hi) / 2
            if (y * np.clip(v - lam * y, 0, self.C)).sum() > 0:
                lo = lam
            else:
                hi = lam
        return np.clip(v - (lo + hi) / 2 * y, 0, self.C)

    # binary dual problem
    def _fit_binary(self, K, y):
        alpha = np.zeros(len(y))
        prev = -np.inf
        for _ in range(self.max_iter):
            grad = 1.0 - y * (K @ (alpha * y)) # d(dual)/d(alpha)
            alpha = self._project(alpha + self.learning_rate * grad, y)
            ay = alpha * y
            obj = alpha.sum() - 0.5 * ay @ K @ ay
            if abs(obj - prev) < self.tol:
                break
            prev = obj

        ay = alpha * y
        sv = alpha > 1e-8
        free = sv & (alpha < self.C - 1e-8)
        idx = free if free.any() else sv
        b = float(np.mean(y[idx] - K[idx] @ ay)) if idx.any() else 0.0
        return alpha, b

    # public
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if len(X) != len(y):
            raise ValueError("X and y are not the same length")

        X_ = np.asarray(X, dtype=float)
        y_ = np.asarray(y).ravel()
        self.classes_ = np.unique(y_)
        self._g = self._gamma_value(X_)
        K = self._kernel(X_, X_)

        # binary: one model (classes_[1] is +1); multiclass: one model per class
        targets = [self.classes_[1]] if len(self.classes_) == 2 else list(self.classes_)
        self.models_ = []
        for c in targets:
            yb = np.where(y_ == c, 1.0, -1.0)
            alpha, b = self._fit_binary(K, yb)
            sv = alpha > 1e-8
            self.models_.append({
                "alpha": alpha[sv], "y": yb[sv], "sv": X_[sv],
                "support": np.where(sv)[0], "b": b,
            })

        first = self.models_[0]
        binary = len(self.models_) == 1
        self.alpha_ = first["alpha"] if binary else [m["alpha"] for m in self.models_]
        self.b_ = first["b"] if binary else [m["b"] for m in self.models_]
        self.support_ = first["support"]
        self.support_vectors_ = first["sv"]
        return self

    def decision_function(self, X):
        X_ = np.asarray(X, dtype=float)
        # f(x) = sum_i alpha_i y_i K(x_i, x) + b
        scores = [self._kernel(X_, m["sv"]) @ (m["alpha"] * m["y"]) + m["b"]
                  for m in self.models_]
        return scores[0] if len(scores) == 1 else np.column_stack(scores)

    def predict(self, X):
        s = self.decision_function(X)
        if len(self.models_) == 1:
            return np.where(s >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(s, axis=1)]

    def save(self, filename=None):
        target = resolve_model_path(filename, "svm")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_class': type(self).__name__,
            'state': {
                'C': self.C,
                'kernel': self.kernel,
                'degree': self.degree,
                'gamma': self.gamma,
                'coef0': self.coef0,
                'max_iter': self.max_iter,
                'learning_rate': self.learning_rate,
                'tol': self.tol,
                'random_state': self.random_state,
                'resolved_gamma': float(self._g),
                'classes': self.classes_.tolist(),
                'models': [
                    {
                        'alpha': m['alpha'].tolist(),
                        'y': m['y'].tolist(),
                        'sv': m['sv'].tolist(),
                        'support': m['support'].tolist(),
                        'b': float(m['b']),
                    }
                    for m in self.models_
                ],
            }
        }
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)
        return target

    @staticmethod
    def load(filename=None):
        target = resolve_model_path(filename, "svm", load=True)
        payload = json.loads(target.read_text(encoding='utf-8'))
        if payload.get('model_class') != 'SVM':
            raise ValueError('Berkas JSON bukan state SVM')

        state = payload.get('state')

        model = SVM(
            C=state.get('C'),
            kernel=state.get('kernel'),
            degree=state.get('degree'),
            gamma=state.get('gamma'),
            coef0=state.get('coef0'),
            max_iter=state.get('max_iter'),
            learning_rate=state.get('learning_rate'),
            tol=state.get('tol'),
            random_state=state.get('random_state'),
        )

        model._g = state.get('resolved_gamma')
        model.classes_ = np.array(state.get('classes'))
        model.models_ = [
            {
                'alpha': np.array(m['alpha'], dtype=float),
                'y': np.array(m['y'], dtype=float),
                'sv': np.array(m['sv'], dtype=float).reshape(len(m['alpha']), -1),
                'support': np.array(m['support'], dtype=int),
                'b': float(m['b']),
            }
            for m in state.get('models')
        ]

        first = model.models_[0]
        binary = len(model.models_) == 1
        model.alpha_ = first['alpha'] if binary else [m['alpha'] for m in model.models_]
        model.b_ = first['b'] if binary else [m['b'] for m in model.models_]
        model.support_ = first['support']
        model.support_vectors_ = first['sv']

        return model
