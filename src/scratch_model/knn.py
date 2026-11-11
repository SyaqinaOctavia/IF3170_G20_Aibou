import numpy as np
import pandas as pd

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
            distance += train_row[i] != test_row[i]
        
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