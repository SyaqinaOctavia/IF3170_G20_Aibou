class NaiveBayes:
    """
    Bayesian Network (Naive Bayes) classifier from scratch, mendukung fitur
    numerik (Gaussian likelihood) dan kategorikal (Categorical likelihood dengan
    Laplace smoothing) secara otomatis per kolom.

    Parameter mengikuti gaya gabungan `sklearn.naive_bayes.GaussianNB` dan
    `sklearn.naive_bayes.CategoricalNB`.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Nilai kecil yang ditambahkan ke varians tiap fitur numerik untuk stabilitas
        numerik (menghindari pembagian dengan nol pada likelihood Gaussian).
    alpha : float, default=1.0
        Parameter Laplace/Lidstone smoothing untuk fitur kategorikal.
    priors : array-like of shape (n_classes,) or None, default=None
        Probabilitas prior tiap kelas. Jika None, akan diestimasi dari data
        (frekuensi relatif kelas pada data training).

    Attributes
    ----------
    classes_ : ndarray
    class_prior_ : ndarray of shape (n_classes,)
    feature_types_ : list of {'numeric', 'categorical'}
    theta_ : dict
        Parameter Gaussian (mean, var) untuk tiap (kelas, fitur numerik).
    cat_prob_ : dict
        Tabel probabilitas kategori untuk tiap (kelas, fitur kategorikal).
    """

    def __init__(self, var_smoothing=1e-9, alpha=1.0, priors=None):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.priors = priors