# ⚽ Football Score Prediction Yooji vs Beeta
## Menyelamatkan Mina dengan Machine Learning

> *"Nyawa Mina berada di ujung jariku. Aku harus memenangkan permainan hidup-mati ini sebelum peluit akhir dibunyikan."*

Submission untuk kompetisi **Tes Ombak**. Notebook ini menyajikan pipeline lengkap untuk memprediksi skor pertandingan sepak bola internasional (`team_goals` dan `opp_goals`) menggunakan pendekatan ensemble tiga model gradient boosting dengan Poisson objective.

## Gambaran Masalah

| Parameter | Detail |
|---|---|
| Data historis | 78.772 baris pertandingan, 30 November 1872 – 4 Agustus 2011 |
| Data prediksi | 42.422 baris, 6 Agustus 2011 – 31 Maret 2026 |
| Target | `team_goals`, `opp_goals` count data, integer non-negatif |
| Metrik evaluasi | Augmented Weighted MAE (AW-MAE) non-linear, penalti berlapis |
| Tantangan utama | Setiap pertandingan = 2 baris (2 perspektif tim) yang harus konsisten satu sama lain |

## Pendekatan

Digunakan **ensemble tiga model komplementer** dengan objective Poisson (cocok untuk count data seperti gol):

| Model | Objective | Kelebihan |
|---|---|---|
| LightGBM | `poisson` | Leaf-wise growth, sangat cepat, efisien memori |
| XGBoost | `count:poisson` | Level-wise growth, regularisasi L1/L2 kuat |
| CatBoost | `Poisson` | Symmetric trees, robust terhadap noise dan outlier |

Setiap model dilatih dua kali (`team_goals` dan `opp_goals`) → **6 model total**. Bobot ensemble dihitung otomatis dari **inverse AW-MAE di validation set** model yang lebih akurat mendapat bobot lebih besar, tanpa tuning manual.

## Metrik Evaluasi: AW-MAE

Kompetisi menggunakan **Augmented Weighted Mean Absolute Error**, metrik non-linear yang lebih ketat dari MAE biasa:

```
AW-MAE = Σ(Loss_i × w_i) / Σw_i
```

### Komponen Penalti

| Komponen | Penalti | Keterangan |
|---|---|---|
| Base MAE | - | Rata-rata selisih absolut skor |
| Exact Score | +0,30 | Jika skor tidak tepat persis |
| Outcome (W/D/L) | +0,25 | Jika hasil akhir salah |
| Goal Difference | +0,15 | Jika selisih gol salah |
| Outcome Multiplier | ×1,5 | Pengali jika outcome salah |
| Non-linear | ^1,5 | Loss total dipangkatkan 1,5 |

### Bobot Turnamen

| Turnamen | Bobot |
|---|---|
| FIFA World Cup | 2,00 |
| UEFA Euro / Copa America / AFC Asian Cup / dll | 1,80 |
| FIFA WC Qualification | 1,60 |
| Friendly | 0,96 |
| Lainnya | 1,20 |

Prediksi `2-1` untuk hasil aktual `0-0` jauh lebih mahal dari sekadar selisih skor outcome (W vs D) juga salah, sehingga mengaktifkan multiplier ×1,5 sekaligus penalti +0,25.

## Struktur Notebook

| Cell | Deskripsi |
|---|---|
| 1 | Instalasi dan import library |
| 2 | Load data dan eksplorasi awal |
| 3 | Implementasi metrik evaluasi AW-MAE |
| 4 | Feature engineering (32 fitur) |
| 5 | Train/validation split |
| 6 | Konfigurasi dan training model ensemble |
| 7 | Validasi dan perhitungan bobot ensemble |
| 8 | Feature importance ensemble |
| 9 | Prediksi ensemble dan consistency post-processing |
| 10 | Analisis distribusi prediksi |
| 11 | Pembuatan file submission Kaggle |
| 12 | Ringkasan pipeline dan rekomendasi pengembangan |

## Data

### Sumber

Tiga file CSV:
- **train.csv** data historis pertandingan 1872-2011, berisi skor aktual sebagai target (78.772 baris × 47 kolom)
- **test.csv** data pertandingan 2011-2026 yang harus diprediksi (42.422 baris × 20 kolom)
- **sample submission.csv** format submission kompetisi

Setiap baris merepresentasikan **satu perspektif tim** dalam satu pertandingan artinya satu pertandingan menghasilkan dua baris (perspektif tim dan perspektif lawan), yang penting untuk post-processing konsistensi di akhir pipeline.

### Feature Engineering

32 fitur final dibagi 8 kelompok:

| Kelompok | Contoh Fitur | Deskripsi |
|---|---|---|
| Team Stats | `team_avg_goals`, `team_avg_conceded`, `team_win_rate` | Statistik historis tim |
| Opp Stats | `opp_avg_goals`, `opp_avg_conceded`, `opp_win_rate` | Statistik historis lawan |
| Recent Form | `recent_attack`, `recent_defense` | Performa 3 tahun terakhir |
| Match Context | `is_home`, `neutral`, `year`, `month`, `gender_enc` | Konteks pertandingan |
| Strength | `attack_vs_defense`, `relative_strength` | Kekuatan relatif tim vs lawan |
| Socioeconomic | `log_gdp_team/opp`, `log_pop_team/opp`, `log_gdp_diff` | Faktor ekonomi-demografi (log1p transform) |
| Geographic | `altitude_venue`, `distance_travel_team/opp`, `temperature_venue` | Faktor geografis venue |
| Confederation | `confederation_team_enc`, `confederation_opp_enc` | Konfederasi (encoded) |

**Mengapa log transformation untuk GDP/populasi?** Distribusinya sangat right-skewed tanpa transformasi, nilai ekstrem negara besar (China, India) akan mendominasi dan membingungkan model.

## Train/Validation Split

90% train (70.894 baris) / 10% validation (7.878 baris), **random split** (bukan time-based) dengan `random_state` tetap untuk reprodusibilitas.

**Mengapa bukan time-based split?** Berbeda dengan time series forecasting, data pertandingan tidak punya dependensi temporal kuat di level baris statistik tim dihitung agregat dari keseluruhan data training. Catatan pengembangan: untuk validasi yang lebih strict, bisa menggunakan pertandingan setelah tahun tertentu sebagai validation set.

## Hasil

### Performa Model & Bobot Ensemble (Validation Set)

| Model | AW-MAE Validasi | Bobot Ensemble |
|---|---|---|
| **LightGBM** | **2,8935** (terbaik) | 33,41% |
| XGBoost | 2,8973 | 33,37% |
| CatBoost | 2,9102 | 33,22% |

Ketiga model punya performa yang sangat berdekatan (selisih AW-MAE < 0,02), sehingga bobot ensemble nyaris merata (~33% masing-masing). Ini menunjukkan ensemble di sini berfungsi lebih sebagai **stabilisasi prediksi (variance reduction)** ketimbang mengandalkan satu model yang jauh lebih superior.

### Feature Importance (LightGBM, berbasis Gain/Split Count)

Fitur paling dominan untuk kedua target (`team_goals` & `opp_goals`): **`year`** jauh di atas fitur lain, diikuti `log_gdp_diff`, `temperature_venue`, `log_gdp_opp`/`log_gdp_team`, `log_pop_opp`, `recent_defense`, dan `relative_strength`.

Dominasi fitur `year` mengindikasikan tren temporal yang kuat pada gaya bermain dan jumlah gol sepanjang 150+ tahun data sepak bola internasional wajar mengingat evolusi taktik, level kompetisi, dan kondisi fisik pemain dari masa ke masa.

### Distribusi Prediksi & Konsistensi

- `pred_team`: rentang [0,070 - 14,201], rata-rata 1,510
- `pred_opp`: rentang [0,067 - 14,590], rata-rata 1,489
- **Consistency check**: 0 inkonsistensi antar pasangan baris pertandingan setelah post-processing
- Submission akhir: 42.422 baris × 3 kolom, tanpa missing value, rentang gol 0–14

## Strategi Ensemble & Post-Processing

**Weighted average dilakukan pada prediksi kontinu (float)**, sebelum dibulatkan ke integer:

```
ŷ_final = w_LGBM · ŷ_LGBM + w_XGB · ŷ_XGB + w_CB · ŷ_CB
```

Rata-rata setelah pembulatan integer akan kehilangan presisi rata-rata `1` dan `2` sama-sama jadi `1.5` → `2`, baik berasal dari `1,4` & `1,6` maupun dari kombinasi lain yang sebenarnya kurang akurat.

**Consistency post-processing**: karena setiap pertandingan punya 2 baris (2 perspektif), prediksi dari kedua perspektif dirata-rata supaya konsisten:

```
goals_A = (pred_A.team + pred_B.opp) / 2
goals_B = (pred_B.team + pred_A.opp) / 2
```

## Instalasi

```bash
pip install lightgbm xgboost catboost
```

Library lain yang digunakan: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## Cara Menjalankan

Jalankan notebook secara berurutan dari Cell 1 hingga Cell 12. Pastikan ketiga file data berada di path yang sesuai (default: `/kaggle/input/competitions/deescuy/`).

```
.
├── notebook.ipynb
├── train.csv
├── test.csv
├── sample submission.csv
└── submission.csv        # dihasilkan otomatis saat Cell 11 dijalankan
```

## Catatan Metodologis

- **Random split, bukan time-based**, untuk validation dipilih karena fitur model berbasis statistik agregat tim, bukan urutan kronologis baris. Pengembangan lanjutan bisa mencoba validasi berbasis tahun untuk evaluasi yang lebih strict.
- **Bobot ensemble nyaris merata (~33% per model)** karena performa AW-MAE ketiga model sangat berdekatan ensemble di sini menambah stabilitas, bukan lompatan akurasi besar dari satu model dominan.
- **Averaging dilakukan pada prediksi float**, bukan setelah dibulatkan, untuk menjaga presisi informasi sebelum konversi ke integer akhir.
- Dominasi fitur `year` pada feature importance perlu diinterpretasi hati-hati bisa mencerminkan tren historis nyata, namun juga berisiko membuat model kurang adaptif jika pola sepak bola modern berubah drastis di luar rentang data training.
