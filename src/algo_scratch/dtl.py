class DecisionTreeClassifier:
    """
    Decision Tree Classifier from scratch menggunakan algoritma C4.5
    (Gain Ratio, mendukung atribut numerik & kategorikal, serta missing value).

    Parameter mengikuti gaya `sklearn.tree.DecisionTreeClassifier`.

    Parameters
    ----------
    criterion : {'gain_ratio', 'entropy'}, default='gain_ratio'
        Kriteria split. 'gain_ratio' adalah kriteria asli C4.5 (Information Gain
        dinormalisasi dengan Split Information). 'entropy' memakai Information
        Gain murni (perilaku seperti ID3) untuk perbandingan.
    max_depth : int or None, default=None
        Kedalaman maksimum pohon. None berarti tumbuh sampai leaf murni atau
        sampai kondisi berhenti lain terpenuhi.
    min_samples_split : int, default=2
        Jumlah sampel minimum pada sebuah node agar node tersebut masih boleh displit.
    min_samples_leaf : int, default=1
        Jumlah sampel minimum yang harus ada pada setiap leaf hasil split.
    min_gain_ratio : float, default=1e-4
        Ambang minimum gain ratio (atau information gain) agar split masih dianggap berguna.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Jumlah fitur yang dipertimbangkan saat mencari split terbaik di setiap node.
    feature_names : list of str or None, default=None
        Nama-nama fitur (opsional, untuk kebutuhan `export_text`/interpretasi).
        Jika None, akan dideteksi otomatis dari DataFrame saat fit, atau diberi nama x0, x1, ...

    Attributes
    ----------
    root_ : _TreeNode
        Root dari pohon keputusan hasil training.
    classes_ : ndarray
        Label kelas unik.
    feature_names_ : list of str
    is_numeric_feature_ : list of bool
        Penanda tipe tiap kolom fitur (numerik / kategorikal) hasil deteksi otomatis.
    """

    def __init__(self, criterion="gain_ratio", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain_ratio=1e-4, max_features=None,
                 feature_names=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.max_features = max_features
        self.feature_names = feature_names
        self.random_state = random_state