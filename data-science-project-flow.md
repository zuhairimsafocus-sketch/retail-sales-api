# Flow Data Science Project (End-to-End hingga MLOps)

Dokumen ini menerangkan aliran kerja lengkap sesebuah projek data science, bermula dari memahami masalah perniagaan sehingga ke peringkat operasi model (MLOps) dalam persekitaran produksi.

> **Nota:** Flow ini bersifat *iterative*, bukan linear. Anda selalunya akan kembali ke fasa terdahulu apabila dapat penemuan baru.

---

## Gambaran Keseluruhan

```
Business        Data           Data            Modeling        Deployment      MLOps
Understanding → Acquisition →  Preparation →   & Evaluation →  & Serving   →  & Monitoring
   (1)            (2)             (3-5)           (6-9)          (10)         (11-13)
        ↑__________________________________________________________________________|
                          (feedback loop / retraining)
```

---

## Notebook → Production Code

Satu prinsip penting dalam flow ini: **eksperimen dalam notebook dahulu, kemudian refactor jadi Python module untuk MLOps.**

Anggap notebook sebagai **makmal** (tempat eksperimen) dan Python files sebagai **kilang** (tempat produksi).

### Bila guna Notebook (.ipynb)

Sesuai untuk kerja *interactive* dan cepat dapat feedback — fokus pada *belajar tentang data*, bukan kod yang kemas:

- **Fasa 3 (EDA):** visualisasi, periksa taburan, korelasi
- **Fasa 4-5 (cleaning & feature engineering):** cuba-cuba transformasi
- **Fasa 6-8 (modeling):** prototaip, bandingkan algoritma, tuning awal

### Bila refactor jadi Python files (.py)

Bila pendekatan dah terbukti berkesan, pindahkan logik ke modul Python yang kemas untuk peringkat produksi (Fasa 10-13).

**Kenapa perlu tukar dari notebook ke .py:**

- Notebook susah nak masuk **version control** (Git) dengan kemas — diff berselerak
- Susah nak buat **unit testing** dan **CI/CD**
- Notebook tak sesuai jalan dalam **pipeline automation** (Airflow, Kubeflow) atau **serving** (FastAPI)
- Kod notebook selalu jalan *out-of-order*, jadi kurang reproducible

### Contoh struktur folder produksi

```
project/
├── notebooks/               # eksperimen & EDA (makmal)
│   ├── 01_eda.ipynb
│   ├── 02_feature_experiments.ipynb
│   └── 03_model_prototyping.ipynb
├── src/                     # production code (kilang)
│   ├── data/
│   │   ├── ingest.py        # ambil & load data
│   │   └── preprocess.py    # cleaning + transformasi
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py         # fungsi latih model
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── pipeline.py          # orchestrate semua
│   └── config.yaml          # parameter & config
├── tests/                   # unit tests
├── api/
│   └── main.py              # FastAPI serving
├── Dockerfile
├── requirements.txt
└── README.md
```

> 💡 **Tip:** Bukan semua kerja notebook dibuang. Sesetengah team simpan notebook EDA sebagai dokumentasi, atau guna `nbconvert` / `jupytext` untuk sync notebook dengan .py.

---

## 1. Business / Problem Understanding

Fasa paling penting tetapi selalu dipandang remeh. Salah faham masalah = projek gagal walaupun model bagus.

- Kenal pasti **masalah perniagaan** yang sebenar (cth: kurangkan churn pelanggan).
- Terjemahkan kepada **masalah ML** (classification? regression? clustering?).
- Tetapkan **success metric**:
  - Metric perniagaan (cth: kurangkan churn 10%)
  - Metric teknikal (cth: F1-score > 0.85)
- Kenal pasti *stakeholders*, kekangan (bajet, masa, latency), dan risiko.
- Tentukan sama ada ML memang diperlukan — kadang-kadang rule-based sudah memadai.

**Output:** Project charter / problem statement yang jelas.

---

## 2. Data Acquisition / Collection

- Kenal pasti sumber data: database (SQL), API, file (CSV/Parquet), web scraping, streaming.
- Pertimbangkan isu **data governance**: kebenaran akses, PII, GDPR/PDPA.
- Wujudkan data pipeline untuk ingest data (cth: Airflow, dbt).
- Versi data anda (cth: **DVC**, LakeFS) supaya boleh reproduce.

**Output:** Raw dataset + skema data yang didokumentasikan.

---

## 3. Exploratory Data Analysis (EDA)

- Fahami struktur, taburan, dan corak data.
- Periksa: missing values, outliers, duplicates, class imbalance.
- Visualisasi: histogram, boxplot, correlation matrix, scatter plot.
- Uji hipotesis awal dan kenal pasti hubungan antara features.

**Tools:** pandas, matplotlib, seaborn, ydata-profiling, Sweetviz.

**Output:** EDA report + senarai insight & isu data.

---

## 4. Data Cleaning / Preprocessing

- Tangani **missing values** (drop, imputation: mean/median/KNN).
- Buang atau betulkan **outliers** dan duplicates.
- Standardkan format (tarikh, teks, unit).
- Tangani **class imbalance** (SMOTE, undersampling, class weights).
- Encode categorical variables (one-hot, label, target encoding).

**Output:** Clean dataset yang sedia untuk feature engineering.

---

## 5. Feature Engineering & Selection

- Cipta features baru (cth: ratio, aggregation, lag features, datetime parts).
- **Scaling/normalization** (StandardScaler, MinMaxScaler) jika perlu.
- **Feature selection**: korelasi, mutual information, feature importance, RFE.
- Pertimbangkan **Feature Store** (Feast) untuk konsistensi train-vs-serve.

> ⚠️ **Penting:** Pastikan tiada *data leakage* — jangan gunakan maklumat masa depan untuk latih model.

**Output:** Feature set akhir + transformation pipeline.

---

## 6. Model Selection & Training

- Pilih algoritma yang sesuai dengan masalah & data:
  - Tabular: Logistic Regression, Random Forest, XGBoost, LightGBM
  - Deep Learning: untuk imej, teks, data kompleks
- Split data: **train / validation / test** (atau cross-validation).
- Latih *baseline model* dahulu sebagai rujukan.
- Track setiap eksperimen (params, metrics, artifacts).

**Tools:** scikit-learn, XGBoost, PyTorch, TensorFlow, **MLflow** / Weights & Biases (tracking).

**Output:** Model terlatih + log eksperimen.

---

## 7. Model Evaluation

- Nilai pada **test set** (data yang model tak pernah lihat).
- Pilih metric yang betul:
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
  - Regression: RMSE, MAE, R²
- Periksa **fairness / bias** merentas kumpulan.
- Analisa ralat (error analysis) — di mana model silap dan kenapa.

**Output:** Evaluation report yang dibanding dengan success metric (Fasa 1).

---

## 8. Hyperparameter Tuning & Optimization

- Tune hyperparameters: Grid Search, Random Search, **Bayesian Optimization** (Optuna).
- Cuba elak overfitting (regularization, early stopping, cross-validation).
- Pertimbangkan trade-off: ketepatan vs latency vs saiz model.

**Output:** Model versi terbaik yang dioptimumkan.

---

## 9. Model Validation & Approval

- Validasi terhadap keperluan perniagaan, bukan hanya metric teknikal.
- Uji robustness (edge cases, adversarial inputs).
- Dokumentasikan model: **Model Card** (tujuan, limitasi, data latihan).
- Dapatkan kelulusan stakeholder sebelum deploy.

**Output:** Model yang diluluskan + dokumentasi.

---

## 10. Deployment / Model Serving

Pilih corak deployment ikut keperluan:

| Corak | Bila guna |
|-------|-----------|
| **Batch prediction** | Ramalan berjadual (cth: harian) |
| **Real-time API** | Latency rendah, on-demand (REST/gRPC) |
| **Streaming** | Data masuk berterusan (Kafka) |
| **Edge** | Pada peranti (mobile, IoT) |

- Bungkus model (FastAPI, BentoML, TorchServe).
- Containerize (**Docker**) dan orchestrate (**Kubernetes**).
- Daftar model dalam **Model Registry** (MLflow Registry).

**Output:** Model yang berjalan di produksi.

---

## 11. MLOps: CI/CD/CT

MLOps menyatukan amalan DevOps dengan kitaran hayat ML.

- **CI (Continuous Integration):** Uji kod, data validation, unit test untuk model.
- **CD (Continuous Delivery):** Auto-deploy pipeline & model ke produksi.
- **CT (Continuous Training):** Auto-retrain model bila ada data baru / prestasi merosot.

**Tools:** GitHub Actions / GitLab CI, **Kubeflow Pipelines**, Airflow, Jenkins.

---

## 12. Monitoring & Observability

Model bukan "deploy and forget" — prestasi merosot dari masa ke masa.

- **Data drift:** taburan input data berubah berbanding data latihan.
- **Concept drift:** hubungan input-output berubah.
- **Model performance:** pantau metric real-time (jika ada ground truth).
- **Operational:** latency, throughput, error rate, kos.
- Tetapkan **alert** apabila metric melebihi ambang.

**Tools:** Evidently AI, Prometheus + Grafana, WhyLabs, Arize.

---

## 13. Retraining & Feedback Loop

- Kumpul ground truth & feedback dari produksi.
- Cetuskan retraining (berjadual atau berdasarkan drift detection).
- Validasi model baru vs model lama (A/B testing, shadow deployment, champion-challenger).
- Promote model baru jika lebih baik → kembali ke Fasa 10.

> Inilah yang menutup *loop* dan menjadikan sistem ML sebagai produk hidup, bukan projek sekali jalan.

---

## Ringkasan Tech Stack (Contoh)

| Peringkat | Tools Popular |
|-----------|---------------|
| Data versioning | DVC, LakeFS |
| EDA & analysis | pandas, seaborn, ydata-profiling |
| Experiment tracking | MLflow, Weights & Biases |
| Feature store | Feast, Tecton |
| Modeling | scikit-learn, XGBoost, PyTorch |
| Pipeline orchestration | Airflow, Kubeflow, Dagster |
| Serving | FastAPI, BentoML, TorchServe |
| Containerization | Docker, Kubernetes |
| CI/CD | GitHub Actions, GitLab CI |
| Monitoring | Evidently, Prometheus, Grafana |

---

## Prinsip Penting Sepanjang Flow

1. **Reproducibility** — sesiapa patut boleh reproduce hasil anda (versi data, kod, environment).
2. **Automation** — automate yang berulang untuk kurangkan ralat manusia.
3. **Versioning** — versi *segala-galanya*: data, kod, model, config.
4. **Collaboration** — gunakan Git, dokumentasi yang baik, code review.
5. **Iteration** — mula kecil (baseline), perbaiki secara berperingkat.

---

*Dokumen rujukan flow data science end-to-end. Sesuaikan ikut keperluan projek dan organisasi anda.*
