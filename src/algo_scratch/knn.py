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

    # ---------------------------------------------------------------------
    def fit(self, X, y):
        """
        Simpan data training. KNN adalah lazy learner: tidak ada proses "training"
        selain menyimpan X dan y.

        TODO:
            - Simpan X sebagai self.X_train_ (ndarray float).
            - Simpan y sebagai self.y_train_ (ndarray).
            - Jika self.task == 'classification', set self.classes_ = label unik dari y.
            - return self
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    def _pairwise_distance(self, X):
        """
        Hitung matriks jarak berbentuk (n_query, n_train) antara setiap baris X
        dan setiap baris self.X_train_, sesuai self.metric (dan self.p jika
        metric='minkowski').

        TODO:
            - metric == 'euclidean' atau (minkowski, p=2) -> Euclidean distance.
              (tips: gunakan identitas ||a-b||^2 = a^2 + b^2 - 2ab agar vektoral/efisien)
            - metric == 'manhattan' atau (minkowski, p=1) -> Manhattan distance.
            - metric == 'chebyshev' -> Chebyshev distance (max |a-b|).
            - metric == 'minkowski' (p umum) -> (sum |a-b|^p) ** (1/p).
            - return array shape (n_query, n_train)
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    def kneighbors(self, X):
        """
        Cari K tetangga terdekat untuk setiap sampel pada X.

        Returns
        -------
        distances : ndarray of shape (n_query, n_neighbors)
            Jarak ke masing-masing K tetangga terdekat, terurut dari yang terdekat.
        indices : ndarray of shape (n_query, n_neighbors)
            Indeks baris pada self.X_train_ / self.y_train_ yang menjadi tetangga.

        TODO:
            - Hitung matriks jarak dengan self._pairwise_distance(X).
            - Untuk tiap baris, ambil n_neighbors indeks dengan jarak terkecil
              (tips: np.argpartition lebih efisien daripada np.argsort penuh,
              lalu urutkan subset tsb).
            - Kembalikan (distances, indices) yang sudah terurut per baris.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    def _get_weights(self, distances):
        """
        Hitung bobot tiap tetangga berdasarkan self.weights.

        TODO:
            - 'uniform'  -> semua bobot = 1 (array sama bentuk dengan `distances`).
            - 'distance' -> bobot = 1 / jarak. Hati-hati pembagian dengan nol
              (gunakan epsilon kecil untuk jarak == 0).
            - selain itu -> raise ValueError.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    def predict(self, X):
        """
        Prediksi label (classification) atau nilai (regression) untuk setiap
        sampel pada X.

        TODO:
            - Dapatkan (distances, indices) via self.kneighbors(X).
            - Dapatkan bobot via self._get_weights(distances).
            - Ambil label/nilai tetangga: self.y_train_[indices].
            - Jika task == 'regression': kembalikan weighted average
              (sum(label * w) / sum(w)) per baris.
            - Jika task == 'classification': lakukan voting berbobot
              (kelas dengan total bobot tertinggi per baris yang menang).
            - return ndarray hasil prediksi.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Kembalikan distribusi probabilitas kelas untuk setiap sampel
        (hanya berlaku untuk task='classification').

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Urutan kolom harus mengikuti urutan self.classes_.

        TODO:
            - raise ValueError jika self.task != 'classification'.
            - Dapatkan (distances, indices) via self.kneighbors(X).
            - Dapatkan bobot via self._get_weights(distances).
            - Untuk tiap sampel, akumulasikan bobot per kelas lalu normalisasi
              (bagi dengan total bobot) agar tiap baris berjumlah 1.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    def score(self, X, y):
        """
        Hitung metrik evaluasi default:
            - classification -> accuracy
            - regression      -> R^2 score

        TODO:
            - y_pred = self.predict(X)
            - Jika regression: hitung R^2 = 1 - SS_res/SS_tot
              (tangani kasus SS_tot == 0).
            - Jika classification: hitung rata-rata (y_pred == y).
        """
        raise NotImplementedError