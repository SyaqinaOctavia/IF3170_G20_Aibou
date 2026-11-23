import json

import numpy as np
import pandas as pd

from ._paths import resolve_model_path

class KNN: #classification saja

    def __init__(self, k_neighbors=5, weights="uniform", p=2, metric="minkowski", task="classification"):
        if k_neighbors < 1:
            raise ValueError("k_neighbors harus >= 1")
    
        if p < 1:
            raise ValueError("p harus >= 1")
    
        if metric not in ["euclidean", "manhattan", "chebyshev", "minkowski"]:
            raise ValueError("Metric tidak tersedia")
        
        self.k_neighbors = k_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric
        self.task = task

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        self.numeric_index = [
            i for i, dtype in enumerate(X.dtypes)
            if pd.api.types.is_numeric_dtype(dtype)
        ]

        self.category_index = [
            i for i, dtype in enumerate(X.dtypes)
            if not pd.api.types.is_numeric_dtype(dtype)
        ]

        return self

    def predict(self, X):
        X_test = np.array(X)
        prediction = []
        for i in range(len(X_test)):
            neigh_index = self.kneighbors(X_test[i])
            nearest_labels = self.y_train[neigh_index]
            
            labels, counts = np.unique(nearest_labels, return_counts=True)
            max_count = np.max(counts)

            candidates = labels[counts == max_count]

            if len(candidates) == 1:
                mode = candidates[0]
            else:
                for idx in neigh_index:
                    if self.y_train[idx] in candidates:
                        mode = self.y_train[idx]
                        break
            prediction.append(mode)

        return np.array(prediction)

    def count_distance(self, train_row, test_row):
        distance, diff = 0, 0

        for i in self.category_index:
            mismatch = train_row[i] != test_row[i]
            if self.metric == "chebyshev":
                distance = max(distance, mismatch)
            else:
                distance += mismatch
        
        for i in self.numeric_index:
            diff = abs(train_row[i] - test_row[i])

            if self.metric == "euclidean":
                distance += diff ** 2
            elif self.metric == "manhattan":
                distance += diff
            elif self.metric == "chebyshev":
                distance = max(distance, diff)
            elif self.metric == "minkowski":
                distance += diff ** self.p
            else:
                raise ValueError(f"Metric '{self.metric}' tidak didukung.")

        if self.metric == "euclidean":
            return np.sqrt(distance)
        elif self.metric == "manhattan" or self.metric == "chebyshev":
            return distance
        elif self.metric == "minkowski":
            return distance ** (1 / self.p)

    def kneighbors(self, new_row):
        distance = []

        for i in range(len(self.X_train)):
            distance.append(self.count_distance(self.X_train[i], new_row))

        distance = np.array(distance)
        nearest_indices = np.argpartition(distance, self.k_neighbors - 1)[:self.k_neighbors]

        return nearest_indices[np.argsort(distance[nearest_indices])]

    def save(self, filename=None):
        target = resolve_model_path(filename, "knn")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_class": type(self).__name__,
            "state": {
                "k_neighbors": self.k_neighbors,
                "weights": self.weights,
                "p": self.p,
                "metric": self.metric,
                "task": self.task,
                "numeric_index": self.numeric_index,
                "category_index": self.category_index,
                "X_train": {"dtype": str(self.X_train.dtype), "data": self.X_train.tolist()},
                "y_train": {"dtype": str(self.y_train.dtype), "data": self.y_train.tolist()},
            },
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        return target

    @staticmethod
    def load(filename=None):
        target = resolve_model_path(filename, "knn", load=True)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("model_class") != "KNN":
            raise ValueError("Berkas JSON bukan state KNN")

        state = payload["state"]
        model = KNN(
            k_neighbors=state["k_neighbors"],
            weights=state["weights"],
            p=state["p"],
            metric=state["metric"],
            task=state["task"],
        )
        model.numeric_index = state["numeric_index"]
        model.category_index = state["category_index"]
        model.X_train = np.array(state["X_train"]["data"], dtype=state["X_train"]["dtype"])
        model.y_train = np.array(state["y_train"]["data"], dtype=state["y_train"]["dtype"])
        return model