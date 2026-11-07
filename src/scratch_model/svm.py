class SVM:
    """
    Support Vector Machine classifier (soft-margin, dual form) from scratch.
    Optimisasi dual Lagrange multipliers (alpha) menggunakan projected gradient
    ascent dengan proyeksi ke constraint box [0, C] dan kendala sum(alpha_i y_i) = 0
    (tanpa memakai solver QP dari library luar -- murni NumPy).

    Untuk multikelas, digunakan strategi One-vs-Rest (OvR), karena SVM dasarnya adalah
    classifier biner.

    Parameter mengikuti gaya `sklearn.svm.SVC`.

    Parameters
    ----------
    C : float, default=1.0
        Parameter regularisasi (soft margin). Semakin besar C, semakin sedikit
        toleransi terhadap pelanggaran margin (semakin mendekati hard-margin).
    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        Jenis kernel yang digunakan.
        - 'linear' : K(x, y) = x . y
        - 'poly'   : K(x, y) = (gamma * x.y + coef0) ** degree
        - 'rbf'    : K(x, y) = exp(-gamma * ||x - y||^2)   (Gaussian kernel)
    degree : int, default=3
        Derajat polinomial (hanya untuk kernel='poly').
    gamma : {'scale', 'auto'} or float, default='scale'
        Koefisien kernel untuk 'rbf' dan 'poly'.
        - 'scale' : gamma = 1 / (n_features * X.var())
        - 'auto'  : gamma = 1 / n_features
    coef0 : float, default=0.0
        Suku independen pada kernel polinomial.
    max_iter : int, default=500
        Jumlah iterasi maksimum untuk projected gradient ascent.
    learning_rate : float, default=1e-3
        Learning rate untuk update alpha.
    tol : float, default=1e-4
        Toleransi konvergensi (berdasar perubahan nilai objektif dual).
    random_state : int or None, default=None

    Attributes
    ----------
    alpha_ : ndarray
        Lagrange multipliers (hanya untuk kasus biner; untuk multikelas disimpan per model OvR).
    support_vectors_ : ndarray
    support_ : ndarray (indeks support vector pada data training)
    b_ : float
        Bias/intercept hasil training.
    classes_ : ndarray
    """

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 max_iter=500, learning_rate=1e-3, tol=1e-4, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state