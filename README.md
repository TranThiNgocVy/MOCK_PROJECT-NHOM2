# Repurchase Prediction — ML Pipeline

> **Bài toán:** Dự đoán xác suất khách hàng **tái mua trong cùng `product_category`** dựa trên lịch sử hành vi e-commerce.  
> **Loại bài toán:** Binary Classification · **Target:** `repurchase` (0 = không tái mua, 1 = tái mua)  
> **Unit of Analysis:** Cặp `(user_id, product_category)` · **Dataset:** 61,728 cặp × 20 cột raw

---

## Mục lục

1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Dữ liệu & Label Generation](#3-dữ-liệu--label-generation)
4. [Quy trình Pipeline](#4-quy-trình-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Kết quả mô hình](#6-kết-quả-mô-hình)
7. [Phân tích kinh doanh](#7-phân-tích-kinh-doanh)
8. [Setup & Chạy](#8-setup--chạy)
9. [Tech Stack](#9-tech-stack)

---

## 1. Tổng quan bài toán

**Bối cảnh kinh doanh:** Nền tảng e-commerce Việt Nam đang đối mặt với tỷ lệ khách không quay lại category cao (~67%). Chi phí acquisition bị lãng phí khi gửi voucher không đúng đối tượng.

**Mục tiêu:** Xây dựng mô hình dự đoán cặp `(user, category)` nào có khả năng tái mua — từ đó tối ưu hóa chiến dịch CRM/voucher.

**Tại sao Binary Classification?**  
Output 0/1 phù hợp để trigger hành động marketing rõ ràng: gửi hoặc không gửi voucher. Mô hình cũng xuất `repurchase_prob` liên tục để hỗ trợ threshold tuning theo chiến lược kinh doanh.

**Tại sao unit of analysis là `(user, category)`?**  
Một người dùng có thể có loyalty khác nhau ở từng category. Aggregate theo `(user, category)` giúp model học category-specific behavior thay vì chỉ nhìn hành vi tổng thể.

**KPI mục tiêu (đặt dựa trên cost analysis):**

| Metric | Cơ sở |
|--------|-------|
| F1-Score | Vượt baseline imbalance (67/33); model phải có giá trị thực tiễn |
| ROC-AUC | Benchmark chuẩn cho bài toán hành vi khách hàng |
| Recall | FN cost đắt gấp 5× FP (−150K vs −30K VND) → ưu tiên bắt đúng |

**```Do chi phí FN cao gấp 5 lần FP, mô hình được đánh giá ưu tiên trên Recall và F1-score nhằm cân bằng giữa khả năng phát hiện khách hàng tái mua và độ chính xác dự đoán.```**

---

## 2. Cấu trúc thư mục

```
mini_project_ml/
│
├── 1_Data_Generation.ipynb            # Khám phá schema, phân phối, label generation
├── 2_Exploratory_Data_Analysis.ipynb  # EDA toàn diện: distribution, correlation, outlier
├── 3_Model_Building.ipynb             # Feature engineering + preprocessing + train 3 models
├── 4_Model_Evaluation.ipynb           # So sánh model, threshold tuning, business analysis
├── bocauhoi.md                        # Bộ câu hỏi phản biện & đáp án chuẩn
├── requirements.txt
└── README.md
│
├── data/
│   ├── synthetic_data.csv             # Dataset gốc: 61,728 dòng × 20 cột (NB1)
│   ├── processed_data.csv             # 13 raw features sau encode + impute + target (14 cột) (NB3)
│   ├── derived_features.csv           # 11 derived features + target (12 cột) (NB3)
│   ├── feature_data.csv               # Dataset cuối: 24 features + target dùng để train (25 cột) (NB3)
│   ├── predictions_lgbm.csv           # Kết quả dự đoán LightGBM trên test set (NB3)
│   ├── predictions_xgb.csv            # Kết quả dự đoán XGBoost trên test set (NB3)
│   └── predictions_lr.csv             # Kết quả dự đoán Logistic Regression trên test set (NB3)
│
└── models/
    ├── best_model.pkl                  # LightGBM — model lưu tại NB3 (⚠ model được chọn cuối: XGBoost — NB4 Bước 13)
    ├── xgb_model.pkl                   # XGBoost — model đã train
    ├── lr_model.pkl                    # Logistic Regression — model đã train
    ├── preprocessor.pkl                # StandardScaler + LabelEncoders (fit trên train set)
    ├── optimal_threshold.pkl           # Threshold tối ưu LightGBM θ* (tìm trên Val set)
    ├── xgb_optimal_threshold.pkl       # Threshold tối ưu XGBoost θ* (tìm trên Val set)
    ├── lr_optimal_threshold.pkl        # Threshold tối ưu LR θ* (tìm trên Val set)
    ├── feature_importance.csv          # Feature importance: LightGBM + XGBoost + Average
    └── test_data.pkl                   # X_test, X_val, y_test, y_val (split 70/15/15)
```

---

## 3. Dữ liệu & Label Generation

### 3.1 Dataset

- **Nguồn:** `pred_repurchase_dataset.csv` — synthetic e-commerce behavioral dataset Việt Nam
- **Kích thước:** 61,728 cặp `(user_id, product_category)` × 20 cột đặc trưng hành vi
- **Key:** `(user_id, product_category)` — mỗi dòng là một cặp duy nhất
- **Đặc điểm:** `cart_rate` có thể > 1 (direct-to-cart, hợp lệ); `avg_rating` có ~28% missing
- **Anti-leakage:** 2 cột bị xóa trước khi xây dựng feature matrix:
  - `purchase_count` — dùng để tạo nhãn (nếu giữ lại = biết đáp án trước)
  - `total_interactions` — = view+click+cart+wishlist+purchase\_count (chứa purchase\_count)

### 3.2 Label Generation — Thuật toán Otsu

Nhãn `repurchase` không có sẵn mà được tạo từ `purchase_count` qua thuật toán Otsu 1D:

$$T^* = \arg\max_T \; \omega_0(T) \cdot \omega_1(T) \cdot [\mu_0(T) - \mu_1(T)]^2$$

Trong đó $\omega_0, \omega_1$ là tỷ lệ hai nhóm và $\mu_0, \mu_1$ là trung bình `purchase_count` của từng nhóm tại ngưỡng $T$.

**Kết quả:** $T^* = 4.02$ → `repurchase = 1` khi `purchase_count ≥ 5`

| Nhãn | Điều kiện | Tỷ lệ |
|------|-----------|-------|
| 0 — Không tái mua | `purchase_count < 5` | **67.2%** |
| 1 — Tái mua | `purchase_count ≥ 5` | **32.8%** |

**Lý do chọn Otsu thay vì đặt ngưỡng tay hay dùng Q2:**  
Otsu xác định ngưỡng dựa trên hình dạng phân phối thực tế — không bias, tối đa hóa inter-class variance, reproducible. Dùng Q2 (median) sẽ cố định 50/50 bất kể phân phối thực.

---

## 4. Quy trình Pipeline

```
pred_repurchase_dataset.csv  (nguồn gốc → lưu vào synthetic_data.csv)
        │
        ▼  NB1 (Bước 1–2) — Data Generation & Understanding
        │   ├─ Bước 1: Đặt vấn đề → business context, định nghĩa bài toán ML
        │   └─ Bước 2: Thu thập dữ liệu → schema, phân phối, label generation (Otsu T*=4.02)
        │
        ▼  NB2 (Bước 3–4) — Exploratory Data Analysis
        │   ├─ Bước 3: Tiền xử lý & EDA → distribution, correlation, outlier, missing
        │   └─ Bước 4: Định nghĩa Target → phân tích class imbalance, đề xuất class_weight/SMOTE
        │
        ▼  NB3 (Bước 5–11) — Model Building
        │   ├─ Bước 5: Feature Engineering → 11 derived features → derived_features.csv
        │   ├─ Bước 6: Feature Selection → feature_data.csv (24 features = 13 raw + 11 derived)
        │   ├─ Bước 7: Train/Val/Test split 70/15/15 + StandardScaler
        │   ├─ Bước 8: Chọn 3 models (LR, XGBoost, LightGBM)
        │   ├─ Bước 9: Huấn luyện với class_weight="balanced"
        │   ├─ Bước 10: Đánh giá baseline → predictions_*.csv
        │   └─ Bước 11: Feature Reduction analysis (PCA)
        │
        ▼  NB4 (Bước 12–15) — Model Evaluation
            ├─ Bước 12: So sánh & Phân tích sâu → ROC/PR Curve, Confusion Matrix
            ├─ Bước 13: Kết luận model tốt nhất → ✅ XGBoost (F1=0.7144, AUC=0.8742)
            ├─ Bước 14: Threshold Optimization → θ* per model, per business scenario
            └─ Bước 15: Business cost analysis → đề xuất 2 kịch bản chiến lược
```

---

## 5. Feature Engineering

### 5.1 Nhóm Raw Features (13 features sau encode/impute)

| Nhóm | Features | Số lượng |
|------|---------|:--------:|
| Demographic | `age` | 1 |
| Behavioral (hành vi tương tác) | `total_view`, `total_click`, `total_cart`, `total_wishlist`, `click_through_rate`, `cart_rate` | 6 |
| Aggregate (tổng hợp giá trị/profile) | `unique_brands`, `avg_discount`, `avg_purchase_value`, `avg_rating`, `user_total_categories`, `category_share` | 6 |
| **Tổng** | | **13** |

> **Features đã loại bỏ (Feature Selection):** `dominant_gender`, `dominant_location`, `avg_price`, `total_interactions` — low importance hoặc anti-leakage.

### 5.2 Nhóm Derived Features (11 features tổng hợp)

| Feature | Mô tả |
|---------|-------|
| `wishlist_to_view_ratio` | total\_wishlist / (total\_view + 1) |
| `wishlist_to_cart_ratio` | total\_wishlist / (total\_cart + 1) |
| `active_engagement_ratio` | Tỷ lệ tương tác chủ động (click + cart) / (pure\_beh + 1) |
| `click_engagement_rate` | (total\_click + total\_cart) / (total\_view + 1) |
| `category_commitment_score` | Điểm cam kết với category (category\_share × pure\_beh) |
| `engagement_depth_score` | Mức độ tương tác sâu (pure\_beh / (user\_total\_categories + 1)) |
| `exploration_score` | Xu hướng khám phá nhiều brand (unique\_brands / (total\_cart + 1)) |
| `brand_loyalty_score` | Điểm trung thành theo brand (total\_cart / (unique\_brands + 1)) |
| `value_per_click` | Giá trị trung bình mỗi lần click |
| `category_breadth_score` | Độ rộng mua sắm trong category |
| `rating_concentration` | Mức độ nhất quán trong đánh giá |

### 5.3 Top Features theo Importance (LightGBM)

| Hạng | Feature | LightGBM Importance |
|:----:|---------|:-------------------:|
| 1 | `category_commitment_score` | 1,546 |
| 2 | `category_share` | 1,205 |
| 3 | `engagement_depth_score` | 1,170 |
| 4 | `avg_purchase_value` | 678 |
| 5 | `user_total_categories` | 608 |
| 6 | `rating_concentration` | 515 |
| 7 | `value_per_click` | 435 |
| 8 | `total_wishlist` | 427 |
| 9 | `category_breadth_score` | 402 |
| 10 | `total_view` | 394 |

---

## 6. Kết quả mô hình

**Data split:** 70% Train · 15% Validation · 15% Test  
**Threshold tuning:** Tối ưu F1 trên Validation set riêng biệt, đánh giá cuối trên Test set

### 6.1 So sánh Test Set Performance

| Model | θ* (Val) | Accuracy | Precision | Recall | **F1-Score** | **ROC-AUC** |
|-------|:--------:|:--------:|:---------:|:------:|:------------:|:-----------:|
| Logistic Regression | 0.3926 | 0.7130 | 0.5396 | 0.8427 | 0.6579 | 0.8297 |
| XGBoost | 0.4772 | 0.7612 | 0.5873 | 0.9116 | **0.7144** | 0.8742 |
| LightGBM | 0.4707 | 0.7581 | 0.5828 | 0.9205 | 0.7137 | **0.8749** |

> **Nhận xét:**  
> - XGBoost và LightGBM đều đạt KPI: F1 ≥ 0.72 ✓, ROC-AUC ≥ 0.87 ✓, Recall ≥ 0.68 ✓  
> - Logistic Regression không đạt F1 target (0.6579 < 0.72) — xác nhận bài toán cần non-linear model  
> - XGBoost nhỉnh hơn LightGBM về F1 (0.7144 vs 0.7137); LightGBM nhỉnh hơn về ROC-AUC (0.8749 vs 0.8742)  
> - **✅ Model được chọn: XGBoost** — F1 cao nhất, level-wise boosting với L1/L2 regularization + Early Stopping (NB4 Bước 13)

### 6.2 Classification Report — XGBoost (Test Set, θ = 0.4772)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| 0 — Không tái mua | 0.94 | 0.69 | 0.79 | 6,227 |
| 1 — Tái mua | 0.59 | 0.91 | 0.71 | 3,033 |
| **Macro avg** | **0.77** | **0.80** | **0.75** | **9,260** |

---

## 7. Phân tích kinh doanh

> **Lưu ý:** XGBoost là model được chọn theo F1 (NB4 Bước 13). Tuy nhiên trong business cost analysis, mỗi kịch bản tối ưu threshold riêng → KB1 dùng LightGBM ở θ cao để kiểm soát FP; KB2 dùng XGBoost ở θ thấp để tối đa Recall. Đây là kết quả tối ưu hóa tổng chi phí kinh doanh — không phải chỉ riêng F1.

**Mô hình chi phí:**  
- **FN Cost** (bỏ sót khách tái mua): −150,000 VND/người — mất doanh thu tiềm năng  
- **FP Cost** (gửi voucher nhầm): −30,000 VND/voucher — lãng phí ngân sách marketing  
- → FN đắt gấp **5×** FP, cần Recall cao để hạn chế bỏ sót

### Kịch bản 1 — Vận hành bền vững hằng ngày

**Model:** LightGBM · **Threshold:** θ = 0.5041

| Chỉ tiêu | Giá trị |
|----------|---------|
| FN Cost — Mất doanh thu | 45.0 triệu VND |
| FP Cost — Voucher lãng phí | 55.9 triệu VND |
| **Tổng thiệt hại** | **100.9 triệu VND** |

Phù hợp: vận hành thường ngày, cần kiểm soát ngân sách marketing chặt, ROI ổn định.

### Kịch bản 2 — Chiến dịch kích cầu mùa cao điểm

**Model:** XGBoost · **Threshold:** θ = 0.3892 (hạ threshold để ép Recall)

| Chỉ tiêu | Giá trị |
|----------|---------|
| FN Cost — Mất doanh thu | 29.7 triệu VND (−15.3M so với KB1) |
| FP Cost — Voucher lãng phí | 64.8 triệu VND (+8.9M so với KB1) |
| **Tổng thiệt hại** | **94.5 triệu VND** |

Phù hợp: flash sale, Tết, 11.11, Black Friday — khi CLV tăng và "thà tốn thêm voucher còn hơn bỏ sót doanh thu".

> **Khuyến nghị:** Dùng Kịch bản 1 (LightGBM) làm baseline hằng ngày. Kích hoạt Kịch bản 2 (XGBoost) trong ±7 ngày quanh mùa lễ lớn. Pipeline hỗ trợ hot-swap model bằng cách thay file `.pkl` mà không cần retrain.

---

## 8. Setup & Chạy

### Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### Thứ tự chạy notebook

| Bước | Notebook | Đầu ra chính |
|:----:|----------|-------------|
| 1 | `1_Data_Generation.ipynb` | Hiểu schema, phân phối target, `synthetic_data.csv` |
| 2 | `2_Exploratory_Data_Analysis.ipynb` | EDA: correlation heatmap, outlier, missing analysis |
| 3 | `3_Model_Building.ipynb` | `feature_data.csv`, `*.pkl` models, `predictions_*.csv` |
| 4 | `4_Model_Evaluation.ipynb` | Bảng so sánh metrics, ROC/PR curve, business scenarios |

> **Lưu ý:** Mỗi notebook import artifacts từ notebook trước. Chạy đúng thứ tự 1 → 4.

---

## 9. Tech Stack

| Thành phần | Thư viện |
|------------|---------|
| Data processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Imbalanced data | `imbalanced-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical analysis | `scipy` |
| Model serialization | `joblib` |
| Environment | Python 3.9+ · Jupyter Notebook / VS Code |
