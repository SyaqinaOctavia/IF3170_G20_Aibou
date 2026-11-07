class KNN:
    """
    K-Nearest Neighbors classifier/regressor from scratch.

    Parameter mengikuti gaya `sklearn.neighbors.KNeighborsClassifier`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Jumlah tetangga terdekat yang digunakan (K).
    weights : {'uniform', 'distance'}, default='uniform'
        Skema pembobotan suara tetangga.
        - 'uniform'  : semua tetangga punya bobot sama.
        - 'distance' : bobot = 1 / jarak (tetangga lebih dekat lebih berpengaruh).
    p : int, default=2
        Parameter power untuk Minkowski distance.
        p=1 -> Manhattan distance, p=2 -> Euclidean distance.
    metric : {'minkowski', 'euclidean', 'manhattan', 'chebyshev'}, default='minkowski'
        Metrik jarak yang digunakan.
    task : {'classification', 'regression'}, default='classification'
        Jenis tugas yang dikerjakan.

    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
    y_train_ : ndarray of shape (n_samples,)
    classes_ : ndarray
        Label kelas unik (hanya untuk classification, diisi di fit()).
    """

    def __init__(self, n_neighbors=5, weights="uniform", p=2, metric="minkowski",
                 task="classification"):
        if n_neighbors < 1:
            raise ValueError("n_neighbors harus >= 1")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric
        self.task = task