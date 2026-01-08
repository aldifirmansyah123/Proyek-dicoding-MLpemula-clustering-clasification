# Submission Akhir BMLP — Clustering & Klasifikasi (Bank Transactions)
Bank Customer Analytics Using Transaction Data (Clustering &amp; Classification) Customer Segmentation and Classification from Banking Transaction Data 

Repositori ini berisi dua tahap utama:
1. **Clustering (Unsupervised Learning)** untuk melakukan segmentasi data transaksi bank tanpa label menggunakan **KMeans**.
2. **Klasifikasi (Supervised Learning)** untuk memprediksi **label cluster** hasil clustering menggunakan beberapa model klasifikasi.

---

## Struktur File

- `[Clustering]_Submission_Akhir_BMLP_Muchamad_Aldi_Firmansyah_(Updated).ipynb`  
  Notebook proses EDA → preprocessing → clustering (KMeans) → evaluasi → feature selection (RFE) → ekspor hasil cluster.

- `[Klasifikasi]_Submission_Akhir_BMLP_Muchamad_Aldi_Firmansyah.ipynb`  
  Notebook klasifikasi untuk memprediksi kolom `Cluster` dari dataset hasil clustering.

- `bank_transactions_data_init.csv`  
  Dataset awal (tanpa label) untuk proses clustering.

- `bank_transactions_data_clustered.csv`  
  Dataset hasil clustering (sudah ada kolom `Cluster`) dan digunakan pada tahap klasifikasi.

---

## Dataset

### 1) Dataset awal: `bank_transactions_data_init.csv`
- Ukuran: **2512 baris × 16 kolom**
- Contoh kolom:
  - `TransactionAmount`, `TransactionDate`, `PreviousTransactionDate`
  - `TransactionType`, `Location`, `Channel`
  - `CustomerAge`, `CustomerOccupation`
  - `TransactionDuration`, `AccountBalance`, dll.

### 2) Dataset hasil clustering: `bank_transactions_data_clustered.csv`
- Ukuran: **2512 baris × 6 kolom**
- Kolom:
  - `AccountBalance`, `CustomerAge` (numerik)
  - `Channel`, `TransactionType`, `Location` (kategorikal)
  - `Cluster` (label hasil KMeans, nilai 0–2)

Distribusi cluster:
- Cluster 0: **932**
- Cluster 1: **845**
- Cluster 2: **735**

---

## Metodologi

### A. Clustering (Notebook: `[Clustering]_...ipynb`)
Alur utama:
1. **EDA** (distribusi fitur numerik & kategorikal, visualisasi, dll.)
2. **Preprocessing**
   - cek duplikasi & missing values
   - konversi kolom tanggal ke datetime
   - drop beberapa kolom yang tidak dipakai (ID/metadata)
   - feature engineering `DaysBetweenTransaction` (selisih tanggal transaksi)
   - handling outlier (IQR, dengan pengecualian fitur tertentu)
   - standardisasi fitur numerik (StandardScaler)
   - label encoding fitur kategorikal (LabelEncoder)
3. **Model Clustering**
   - Menentukan jumlah cluster optimal (Elbow Method)
   - Melatih **KMeans** dengan **k = 3**
4. **Evaluasi**
   - Silhouette Score (sebelum feature selection): **0.4894**
5. **Feature Selection (RFE - opsional)**
   - Memilih 5 fitur terbaik:
     - `AccountBalance`, `CustomerAge`, `Channel`, `TransactionType`, `Location`
   - Silhouette Score (sesudah feature selection): **0.5360**
6. **Ekspor**
   - Output disimpan sebagai `bank_transactions_data_clustered.csv`

Insight ringkas (berdasarkan analisis notebook):
- Rata-rata `AccountBalance` dan `CustomerAge` antar cluster relatif mirip.
- Perbedaan cluster paling terlihat dari **dominasi `Location`** (masing-masing cluster punya lokasi modus berbeda).

---

### B. Klasifikasi (Notebook: `[Klasifikasi]_...ipynb`)
Tujuan: memprediksi label `Cluster` dari fitur pada dataset hasil clustering.

Langkah:
1. Load `bank_transactions_data_clustered.csv`
2. Encoding fitur kategorikal (`Channel`, `TransactionType`, `Location`)
3. Normalisasi fitur numerik (MinMaxScaler)
4. Split data 80:20 (random_state=42)
5. Training beberapa model:
   - KNN
   - Decision Tree
   - Random Forest
   - SVM
   - Naive Bayes
6. Evaluasi (Accuracy, Precision, Recall, F1)

Hasil evaluasi (test set):
| Model | Accuracy | F1 (weighted) |
|------|----------|---------------|
| KNN | 0.9344 | 0.9347 |
| Decision Tree | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 |
| SVM | 0.9781 | 0.9781 |
| Naive Bayes | 1.0000 | 1.0000 |

Observasi:
- **Decision Tree, Random Forest, dan Naive Bayes** mencapai skor sempurna pada semua metrik.
- **SVM** sangat baik (≈97.8%).
- **KNN** cukup baik (≈93.4%), namun di bawah model terbaik.

---

## Requirement

Disarankan menggunakan Python 3.9+.

Library yang dipakai di notebook:
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- yellowbrick

Install cepat:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick


Cara Menjalankan

Catatan: Notebook menggunakan path Colab /content/.... Jika dijalankan lokal, ubah path menjadi relatif, misalnya ./bank_transactions_data_init.csv.

1) Jalankan Clustering

Buka dan run semua cell pada:

[Clustering]_Submission_Akhir_BMLP_Muchamad_Aldi_Firmansyah_(Updated).ipynb

Output:

bank_transactions_data_clustered.csv

2) Jalankan Klasifikasi

Buka dan run semua cell pada:

[Klasifikasi]_Submission_Akhir_BMLP_Muchamad_Aldi_Firmansyah.ipynb

Input:

bank_transactions_data_clustered.csv

Catatan Reproducibility

KMeans menggunakan random_state=0.

Label cluster (0/1/2) bisa saja tertukar urutannya jika konfigurasi/seed berubah, meskipun struktur cluster sama.

Author

Muchamad Aldi Firmansyah
