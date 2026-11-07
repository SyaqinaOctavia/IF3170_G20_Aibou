class LogisticRegression:
    """
    Logistic Regression from scratch, dioptimasi dengan (batch/mini-batch) gradient
    descent, mendukung klasifikasi biner maupun multikelas (strategi One-vs-Rest).

    Parameter mengikuti gaya `sklearn.linear_model.LogisticRegression`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Jenis regularisasi yang digunakan.
    C : float, default=1.0
        Inverse dari kekuatan regularisasi (seperti di sklearn). Semakin kecil C,
        semakin kuat regularisasinya. Secara internal, lambda = 1 / C.
    l1_ratio : float, default=0.5
        Hanya dipakai jika penalty='elasticnet'. Proporsi campuran L1 (0 <= l1_ratio <= 1).
        l1_ratio=1 -> murni L1, l1_ratio=0 -> murni L2.
    fit_intercept : bool, default=True
        Apakah menambahkan bias/intercept.
    max_iter : int, default=1000
        Jumlah iterasi maksimum gradient descent.
    learning_rate : float, default=0.1
        Learning rate untuk gradient descent.
    tol : float, default=1e-6
        Toleransi konvergensi; optimisasi berhenti bila perubahan loss < tol.
    solver : {'gradient_descent'}, default='gradient_descent'
        Metode optimisasi (hanya gradient descent yang diimplementasikan, from scratch).
    batch_size : int or None, default=None
        Jika None -> full-batch gradient descent. Jika diisi -> mini-batch gradient descent.
    random_state : int or None, default=None

    Attributes
    ----------
    coef_ : ndarray of shape (n_classes, n_features) atau (n_features,) untuk biner
    intercept_ : ndarray of shape (n_classes,) atau skalar untuk biner
    classes_ : ndarray
    loss_history_ : list of float
    """

    def __init__(self, penalty="l2", C=1.0, l1_ratio=0.5, fit_intercept=True,
                 max_iter=1000, learning_rate=0.1, tol=1e-6,
                 solver="gradient_descent", batch_size=None, random_state=None):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.solver = solver
        self.batch_size = batch_size
        self.random_state = random_state