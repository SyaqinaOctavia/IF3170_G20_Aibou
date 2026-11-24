# Tugas Besar 2 IF3170 Inteligensi Artifisial — Kelompok 20 (Aibou)

Repository ini berisi pipeline klasifikasi biner untuk memprediksi **pembatalan reservasi hotel** (`is_canceled`) berdasarkan karakteristik pemesanan, seperti jumlah tamu, lama menginap, jenis kamar, kanal pemesanan, harga per malam, dan riwayat tamu. Lima algoritma klasifikasi diimplementasikan **dari nol (*from scratch*)** menggunakan NumPy/Pandas, lalu dievaluasi dan dibandingkan dengan implementasi `scikit-learn`.

Disusun oleh:

| Nama | NIM |
| --- | --- |
| Faiq Azzam Nafidz | 13524003 |
| Mikhael Benrael Tampubolon | 13524009 |
| Muhammad Nafis Habibi | 13524018 |
| Syaqina Octavia Rizha | 13524088 |
| Muhammad Rafi Akbar | 13524128 |

## Daftar Isi

- [Deskripsi Singkat](#deskripsi-singkat)
- [Struktur Repository](#struktur-repository)
- [Setup](#setup)
- [Cara Menjalankan Program](#cara-menjalankan-program)
- [Menggunakan Model *From-Scratch* Secara Langsung](#menggunakan-model-from-scratch-secara-langsung)
- [Output](#output)
- [Pembagian Tugas](#pembagian-tugas)

## Deskripsi Singkat

| Aspek | Keterangan |
| --- | --- |
| **Masalah** | Klasifikasi biner: apakah suatu reservasi hotel akan dibatalkan (`1`) atau tidak (`0`) |
| **Dataset** | `train.csv` (27.373 baris, berlabel) dan `test.csv` (6.844 baris, tanpa label) |
| **Metrik** | Macro F1-Score |
| **Algoritma (from scratch)** | K-Nearest Neighbors, Decision Tree Learning, Logistic Regression, Support Vector Machine, Naive Bayes |
| **Pembanding** | Implementasi `scikit-learn` dengan hyperparameter yang sama |

Alur kerja pada notebook `src/main.ipynb`:

1. **Initialization** — import library, pengaturan seed, dan *loading* data.
2. **Data Understanding** — EDA: struktur data, statistik, korelasi, missing value, dan hubungan fitur kategorikal terhadap target.
3. **Training and Validation Split** — pembagian 80:20 dengan *stratified sampling*.
4. **Data Preparation** — pembersihan data dan *feature engineering* (fitur temporal, indikator missing value, total malam, estimasi biaya, rasio pembatalan sebelumnya, dll.).
5. **Preparation Pipeline** — `ColumnTransformer` + `Pipeline` (imputasi, *scaling*, *one-hot encoding*).
6. **Modeling and Validation** — *tuning* hyperparameter dengan `GridSearchCV` (3-fold stratified CV), lalu pelatihan dan evaluasi kelima model *from-scratch*.
7. **Comparison with scikit-learn** — perbandingan Macro F1 model *from-scratch* vs `scikit-learn`.
8. **Saving and Loading Models** — penyimpanan/pemuatan model (format JSON) dan preprocessor (pickle).
9. **Kaggle Submission** — pelatihan ulang model terbaik pada seluruh data latih dan pembuatan `submission.csv`.
10. **Error Analysis** — *confusion matrix* dan analisis kesalahan prediksi.

## Struktur Repository

```
IF3170_G20_Aibou/
├── docs/                          # Dokumen/laporan
├── input/
│   ├── train.csv                  # Data latih (berlabel)
│   ├── test.csv                   # Data uji (tanpa label)
│   └── sample_submission.csv      # Contoh format submission Kaggle
├── output/
│   ├── saved_models/              # Model tersimpan (.json) dan preprocessor.pkl
│   └── submission.csv             # Dihasilkan oleh notebook
└── src/
    ├── main.ipynb                 # Notebook utama (EDA → modeling → submission)
    └── scratch_model/             # Implementasi algoritma from scratch
        ├── __init__.py
        ├── _paths.py              # Utilitas path penyimpanan model
        ├── knn.py                 # K-Nearest Neighbors
        ├── dtl.py                 # Decision Tree Learning
        ├── logistic_regression.py # Logistic Regression
        ├── logistic_regression_visualization.py
        ├── svm.py                 # Support Vector Machine
        └── naive_bayes.py         # Naive Bayes
```

## Setup

### Prasyarat

- Python 3.10 atau lebih baru (kode memakai sintaks tipe seperti `tuple[...]` dan `X | None`)
- `pip`
- (Opsional) [Graphviz](https://graphviz.org/download/) jika ingin memvisualisasikan pohon keputusan

### Langkah Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/<username>/IF3170_G20_Aibou.git
   cd IF3170_G20_Aibou
   ```

2. **Buat dan aktifkan virtual environment**

   ```bash
   python -m venv venv

   # Linux / macOS
   source venv/bin/activate

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   ```

3. **Install dependensi**

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn graphviz jupyter
   ```

4. **Pastikan data tersedia** di folder `input/` (`train.csv`, `test.csv`, dan `sample_submission.csv`).

## Cara Menjalankan Program

1. Jalankan Jupyter dari root repository:

   ```bash
   jupyter notebook
   ```

   Atau, jika memakai VS Code, buka folder repository lalu pilih kernel Python dari virtual environment tadi.

2. Buka `src/main.ipynb`.

3. Jalankan seluruh sel secara berurutan (**Run All**). Notebook akan otomatis:
   - mendeteksi *root* proyek dan menambahkan `src/` ke `sys.path`,
   - membuat folder `output/` dan `output/saved_models/` bila belum ada,
   - melatih dan mengevaluasi kelima model,
   - menyimpan model terbaik, serta menghasilkan `output/submission.csv`.

## Menggunakan Model *From-Scratch* Secara Langsung

Semua model dapat diimpor dari paket `scratch_model` (jalankan dari folder `src/`) dan memiliki antarmuka `fit` / `predict` / `save` / `load`:

```python
import numpy as np
from scratch_model import KNN, DecisionTreeClassifier, LogisticRegression, SVM, NaiveBayes

# X_train, y_train, X_val: data yang sudah melalui preprocessing
model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, learning_rate=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Simpan ke output/saved_models/<nama>.json
model.save("my_logreg")

# Muat kembali
model = LogisticRegression.load("my_logreg")
```

Jika nama file tidak diberikan, model disimpan dengan penomoran otomatis (`<nama>_1.json`, `<nama>_2.json`, ...), dan `load()` tanpa argumen akan memuat versi terbaru.

## Output

- `output/submission.csv`: Prediksi akhir pada `test.csv` (kolom `id`, `is_canceled`) untuk dikumpulkan ke Kaggle
- `output/saved_models/*.json`: Parameter model *from-scratch* yang telah dilatih
- `output/saved_models/preprocessor.pkl`: Preprocessor (`ColumnTransformer`) yang digunakan saat pelatihan

## Pembagian Tugas

| Nama | NIM | Tugas |
| --- | --- | --- |
| Faiq Azzam Nafidz | 13524003 | Implementasi SVM, Laporan |
| Mikhael Benrael Tampubolon | 13524009 | Implementasi DTL, Laporan |
| Muhammad Nafis Habibi | 13524018 | Implementasi Naive Bayes, Laporan |
| Syaqina Octavia Rizha | 13524088 | Implementasi KNN, Pengisian Notebook, Laporan |
| Muhammad Rafi Akbar | 13524128 | Implementasi Logistic Regression, Laporan |