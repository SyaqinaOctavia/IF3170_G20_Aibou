import json # gak perlu di deepnote 
import math # gak perlu di deepnote 

import numpy as np # gak perlu di deepnote 

from ._paths import resolve_model_path # gak perlu di deepnote 

class LogisticRegression:

    def __init__(self, penalty="l2", C=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, learning_rate=0.1, tol=1e-6, solver="gradient_descent", batch_size=None, random_state=None, record_history=False):
        if penalty not in ("l1", "l2", "elasticnet", None):
            raise ValueError("penalty harus 'l1', 'l2', 'elasticnet', atau None")
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C harus berupa angka valid yang lebih besar dari 0")
        if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio harus berupa angka valid di antara 0 dan 1")
        if max_iter < 1 or not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(tol) or tol < 0:
            raise ValueError("max_iter dan learning_rate harus positif; tol harus valid dan tidak negatif")
        if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size harus berupa bilangan bulat positif atau None")
        if solver != "gradient_descent":
            raise ValueError("solver yang didukung hanya 'gradient_descent'")
        
        self.penalty = penalty
        self.C = float(C)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.tol = float(tol)
        self.solver = solver
        self.batch_size = batch_size
        self.random_state = random_state
        self.record_history = bool(record_history)

    @staticmethod
    def _encode_json(value):
        if isinstance(value, np.ndarray):
            return {"__jenis__": "array", "dtype": str(value.dtype),
                    "data": LogisticRegression._encode_json(value.tolist())}
        if isinstance(value, np.generic):
            return LogisticRegression._encode_json(value.item())
        if isinstance(value, dict):
            return {"__jenis__": "dict", "data": [
                [LogisticRegression._encode_json(k), LogisticRegression._encode_json(v)]
                for k, v in value.items()
            ]}
        if isinstance(value, tuple):
            return {"__jenis__": "tuple", "data": [LogisticRegression._encode_json(v) for v in value]}
        if isinstance(value, list):
            return [LogisticRegression._encode_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            marker = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
            return {"__jenis__": "float", "data": marker}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Tipe {type(value).__name__} belum didukung untuk penyimpanan JSON")

    @staticmethod
    def _decode_json(value):
        if isinstance(value, list):
            return [LogisticRegression._decode_json(v) for v in value]
        if isinstance(value, dict):
            kind = value.get("__jenis__")
            if kind == "array":
                return np.asarray(LogisticRegression._decode_json(value["data"]), dtype=value["dtype"])
            if kind == "tuple":
                return tuple(LogisticRegression._decode_json(v) for v in value["data"])
            if kind == "dict":
                return {LogisticRegression._decode_json(k): LogisticRegression._decode_json(v)
                        for k, v in value["data"]}
            if kind == "float":
                return {"nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}[value["data"]]
            return {k: LogisticRegression._decode_json(v) for k, v in value.items()}
        return value

    def save(self, filename=None):
        target = resolve_model_path(filename, "lr")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "model_class": type(self).__name__,
            "state": self._encode_json(self.__dict__),
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, filename=None):
        target = resolve_model_path(filename, "lr", load=True)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("model_class") != cls.__name__:
            raise ValueError("Berkas JSON bukan state Logistic Regression yang didukung")
        model = cls.__new__(cls)
        model.__dict__.update(cls._decode_json(payload["state"]))
        return model

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _regularization_loss(self, coef):
        strength = 1.0 / (self.C * self.n_samples_fit_)
        if self.penalty == "l1":
            return strength * np.abs(coef).sum()
        if self.penalty == "l2":
            return 0.5 * strength * np.square(coef).sum()
        if self.penalty == "elasticnet":
            return strength * (self.l1_ratio * np.abs(coef).sum()
                               + 0.5 * (1.0 - self.l1_ratio) * np.square(coef).sum())
        return 0.0

    def _loss(self, X, targets, theta):
        scores = X @ theta.T
        # BCE langsung dari logits 
        data_loss = np.mean(np.logaddexp(0.0, scores) - targets * scores)
        coef = theta[:, :self.n_features_in_]
        return data_loss + self._regularization_loss(coef) / max(1, theta.shape[0])

    def fit(self, X, y):
        feature_names = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2 or len(X) == 0 or len(X) != len(y):
            raise ValueError("X harus berupa matriks 2D tidak kosong dengan satu label per baris")
        if not np.isfinite(X).all():
            raise ValueError("X hanya boleh berisi angka valid, handle missing value duluan")
        if any(v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in y):
            raise ValueError("y tidak boleh berisi label yang hilang")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("y harus memiliki minimal dua kelas")
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = len(X)
        self.feature_names_in_ = feature_names
        self.is_binary_ = len(self.classes_) == 2
        targets = (y == self.classes_[1]).astype(float).reshape(-1, 1) if self.is_binary_ else np.column_stack([(y == c).astype(float) for c in self.classes_])
        if self.fit_intercept:
            X_fit = np.column_stack([X, np.ones(len(X))])
        else:
            X_fit = X
        n_outputs = targets.shape[1]
        theta = np.zeros((n_outputs, X_fit.shape[1]), dtype=float)
        theta_history = [theta.copy()] if self.record_history else None
        rng = np.random.default_rng(self.random_state)
        batch_size = len(X) if self.batch_size is None else min(self.batch_size, len(X))
        self.loss_history_ = []

        for _ in range(self.max_iter):
            order = rng.permutation(len(X_fit)) if batch_size < len(X_fit) else np.arange(len(X_fit))
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X_fit[idx], targets[idx]
                probs = self._sigmoid(xb @ theta.T)
                grad = ((probs - yb).T @ xb) / (len(idx) * n_outputs)
                coef_count = self.n_features_in_
                strength = 1.0 / (self.C * self.n_samples_fit_)
                if self.penalty == "l2":
                    grad[:, :coef_count] += strength * theta[:, :coef_count] / n_outputs
                elif self.penalty == "elasticnet":
                    grad[:, :coef_count] += (strength * (1.0 - self.l1_ratio)
                        * theta[:, :coef_count] / n_outputs)
                theta -= self.learning_rate * grad

                # Proximal soft-thresholding utk handle penalti L1 dengan lebih stabil.
                l1_fraction = 1.0 if self.penalty == "l1" else (self.l1_ratio if self.penalty == "elasticnet" else 0.0)
                if l1_fraction:
                    threshold = self.learning_rate * strength * l1_fraction / n_outputs
                    coef = theta[:, :coef_count]
                    theta[:, :coef_count] = np.sign(coef) * np.maximum(np.abs(coef) - threshold, 0.0)
            if theta_history is not None:
                theta_history.append(theta.copy())
            loss = self._loss(X_fit, targets, theta)
            self.loss_history_.append(float(loss))
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-2] - loss) < self.tol:
                break

        self.n_iter_ = len(self.loss_history_)
        if theta_history is not None:
            self.theta_history_ = np.asarray(theta_history)
        self.coef_matrix_ = theta[:, :self.n_features_in_].copy()
        self.intercept_vector_ = theta[:, -1].copy() if self.fit_intercept else np.zeros(n_outputs)
        if self.is_binary_:
            self.coef_ = self.coef_matrix_[0].copy()
            self.intercept_ = float(self.intercept_vector_[0])
        else:
            self.coef_ = self.coef_matrix_.copy()
            self.intercept_ = self.intercept_vector_.copy()
        return self

    def val_x(self, X):
        feature_names = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X harus memiliki jumlah fitur yang sama seperti saat fit")
        if not np.isfinite(X).all():
            raise ValueError("X hanya boleh berisi angka valid, handle missing value duluan")
        if self.feature_names_in_ is not None and feature_names is not None and feature_names != self.feature_names_in_:
            raise ValueError("Nama dan urutan fitur harus sama seperti data training")
        return X

    def decision(self, X):
        X = self.val_x(X)
        scores = X @ self.coef_matrix_.T + self.intercept_vector_
        return scores[:, 0] if self.is_binary_ else scores

    def prediction_probability(self, X):
        scores = self.decision(X)
        if self.is_binary_:
            p1 = self._sigmoid(scores)
            return np.column_stack([1.0 - p1, p1])
        probs = self._sigmoid(scores)
        return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-15, None)

    def predict(self, X):
        scores = self.decision(X)
        if self.is_binary_:
            return self.classes_[(scores >= 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]
