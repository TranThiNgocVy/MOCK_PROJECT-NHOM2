# 📚 Giải Thích Code — Notebook 3 & Notebook 4

> Tài liệu này trả lời 3 câu hỏi:
> 1. Toàn bộ code NB3 + NB4 làm gì?
> 2. NB3: Model bắt đầu học từ đâu? Test set vào từ đâu?
> 3. NB4: Evaluation đã đủ chuẩn khoa học chưa? Cách cải thiện?

---

## 🔷 NOTEBOOK 3 — `3_Model_Building.ipynb`

### Mục đích tổng quát
NB3 là notebook xây dựng mô hình. Nó nhận vào CSV thô, xử lý feature, chia train/test, train 3 model, và lưu model ra file pkl.

---

### Cell-by-cell giải thích

#### 📦 Cell 1 — Import & Setup
```python
import pandas, numpy, matplotlib, sklearn, xgboost, lightgbm, joblib, time
RANDOM_STATE = 42
BASE_DIR / DATA_DIR / MODELS_DIR = Path(...)
```
- Khai báo tất cả thư viện cần dùng
- `RANDOM_STATE = 42` → đảm bảo tái lập được kết quả (reproducibility)
- `MODELS_DIR` là thư mục sẽ lưu file `.pkl`

---

#### 📂 Cell 2 — Load Data
```python
df = pd.read_csv('pred_repurchase_dataset.csv')
# 61,728 dòng × 20 cột
```
- Đọc file CSV đã tạo từ NB1
- `purchase_count` **không có** trong CSV — đã cố ý loại từ NB1 để tránh label leakage trực tiếp

---

#### 🎯 Cell 3 — Feature Selection (đã sửa leakage)
```python
BEHAVIORAL_COLS = ['total_view', 'total_click', 'total_cart', 'total_wishlist',
                   'click_through_rate', 'cart_rate']
# total_interactions ĐÃ BỊ XÓA
ALL_FEATURE_COLS = DEMO_COLS + BEHAVIORAL_COLS + AGGREGATE_COLS  # 16 features
X_raw = df[ALL_FEATURE_COLS].copy()
y     = df['repurchase'].astype(int)
```
- **Tại sao xóa `total_interactions`?**
  - `total_interactions = view + click + cart + wishlist + purchase_count`
  - Nhãn `repurchase = 1` khi `purchase_count >= 5`
  - → `total_interactions - (view+click+cart+wishlist) = purchase_count`
  - → Model chỉ cần nhìn `total_interactions` là biết nhãn → **F1 giả 0.99**
- Kết quả: 16 features thật sự mô tả hành vi user

---

#### 🔧 Cell 4 — Preprocessing
```python
le = LabelEncoder()
X_processed['dominant_gender'] = le.fit_transform(...)
# Impute avg_rating NaN bằng median
X_processed['avg_rating'].fillna(median, inplace=True)
```
- `LabelEncoder` chuyển cột text thành số (Nam→0, Nữ→1; Hà Nội→1, v.v.)
- `avg_rating` bị NaN 28% → impute bằng median (giá trị an toàn nhất cho skewed data)
- **Chú ý**: imputation dùng median của toàn bộ dataset (global) — điểm yếu nhỏ: lý tưởng hơn là impute bằng median của train set sau khi split

---

#### ✂️ Cell 5 — Train/Test Split (80/20)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.20,
    random_state=42,
    stratify=y   # giữ tỷ lệ class 67%/33% trong cả 2 tập
)
# Train: 49,382 | Test: 12,346
```
- `stratify=y` → đảm bảo train và test có cùng tỷ lệ class (không bị mất balanced)
- `random_state=42` → kết quả luôn giống nhau mỗi lần chạy
- **Test set bị "khóa"** từ đây — không được nhìn cho đến khi evaluate

---

#### 📏 Cell 6 — StandardScaler (cho LR)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # học mean/std từ TRAIN ONLY
X_test_scaled  = scaler.transform(X_test)        # áp dụng mean/std của train
```
- LR nhạy cảm với scale → cần normalize
- **Quan trọng**: `fit_transform` chỉ trên train → không bị **data leakage từ scaler**
- XGBoost & LightGBM: không cần scale (tree-based models invariant với monotone transforms)

---

#### 🔨 Cell 7 — Derived Features (7 features mới)
```python
pure_beh = total_view + total_click + total_cart + total_wishlist  # KHÔNG có purchase
d['view_to_click_ratio']       = total_click  / (total_view + 1)
d['cart_to_click_ratio']       = total_cart   / (total_click + 1)
d['wishlist_to_view_ratio']    = total_wishlist / (total_view + 1)
d['active_engagement_ratio']   = (click + cart) / (pure_beh + 1)   # ← dùng pure_beh
d['category_commitment_score'] = category_share × pure_beh          # ← dùng pure_beh
d['exploration_score']         = unique_brands / (total_cart + 1)
d['engagement_depth_score']    = pure_beh / (user_total_categories + 1)  # ← dùng pure_beh
```
- Tạo ra tỉ lệ hành vi (ratio features) → thường mạnh hơn raw counts
- 3 features dùng `pure_beh` thay vì `total_interactions` → tránh leakage cascade

---

#### 🧩 Cell 8 — Full Feature Matrix (23 features)
```python
X_full  = df_featured.drop(columns=['repurchase'])  # 23 features
X_train = X_full.loc[X_train.index]  # giữ đúng index của train
X_test  = X_full.loc[X_test.index]   # giữ đúng index của test
# Scale lại với 23 features
scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train)   # fit on train
X_test_scaled  = scaler_full.transform(X_test)        # transform test
```
- Kết hợp 16 raw + 7 derived = **23 features**
- Dùng `.loc[index]` để đảm bảo đúng dòng cho mỗi tập

---

#### 🤖 Cell 9 — Helper Function `evaluate_model`
```python
def evaluate_model(model, X, y_true, label, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]   # xác suất class = 1
    preds = (probs >= threshold).astype(int)
    # tính accuracy, precision, recall, f1, auc
    print(classification_report(...))
    return metrics_dict, probs
```
- Hàm dùng chung cho cả 3 models
- `predict_proba(X)[:, 1]` → lấy xác suất dự đoán **tái mua** (class=1)
- `threshold=0.5` mặc định

---

### ⭐ CÂU HỎI 2: Model học từ đâu? Test vào từ đâu?

```
┌──────────────────────────────────────────────────────────────────┐
│  MODEL BẮT ĐẦU HỌC (FIT)                                        │
├──────────────────────────────────────────────────────────────────┤
│  Cell #VSC-8005efc4 (LR):                                        │
│    lr_model.fit(X_train_scaled, y_train)  ← ĐÂY LÀ ĐIỂM HỌC   │
│                                                                  │
│  Cell #VSC-6781b019 (XGBoost):                                   │
│    xgb_model.fit(X_train, y_train)        ← ĐÂY LÀ ĐIỂM HỌC   │
│                                                                  │
│  Cell #VSC-275ae592 (LightGBM):                                  │
│    lgbm_model.fit(X_train, y_train)       ← ĐÂY LÀ ĐIỂM HỌC   │
├──────────────────────────────────────────────────────────────────┤
│  TEST SET VÀO LẤY KẾT QUẢ (PREDICT)                             │
├──────────────────────────────────────────────────────────────────┤
│  Cell #VSC-8005efc4 (LR):                                        │
│    evaluate_model(lr_model, X_test_scaled, y_test, ...)          │
│    → predict_proba(X_test_scaled)[:, 1]   ← TEST VÀO ĐÂY       │
│                                                                  │
│  Cell #VSC-36046e9e (XGBoost):                                   │
│    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]            │
│    xgb_preds = (xgb_probs >= 0.5).astype(int)                   │
│                                  ← TEST VÀO ĐÂY                 │
│                                                                  │
│  Cell #VSC-84c91337 (LightGBM):                                  │
│    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]          │
│    lgbm_preds = (lgbm_probs >= optimal_threshold).astype(int)   │
│                                  ← TEST VÀO ĐÂY                 │
└──────────────────────────────────────────────────────────────────┘
```

**Luồng dữ liệu tóm tắt:**
```
CSV (61,728 rows)
    ↓ read_csv
df  →  X_raw (16 features)  +  y (nhãn)
    ↓ preprocessing (encode, impute)
X_processed (16 features, sạch)
    ↓ train_test_split(stratify=y, 80/20)
X_train (49,382)          X_test (12,346) ← KHÓA đến bước cuối
    ↓ add derived features      ↓
X_train (23 features)    X_test (23 features)
    ↓ fit_transform(scaler)     ↓ transform(scaler)
X_train_scaled           X_test_scaled
    ↓                           ↓
model.fit(X_train, y_train)   model.predict(X_test)  ← KẾT QUẢ
```

---

#### 💾 Cell cuối NB3 — Save Models
```python
joblib.dump(lgbm_model, MODELS_DIR / 'best_model.pkl')
joblib.dump({'X_test': X_test, 'X_test_lr': X_test_scaled, 'y_test': y_test},
            MODELS_DIR / 'test_data.pkl')
```
- Lưu model + test data để NB4 load lại và đánh giá độc lập

---

## 🔶 NOTEBOOK 4 — `4_Model_Evaluation.ipynb`

### Mục đích tổng quát
NB4 là notebook đánh giá model độc lập — **không train lại**, chỉ load model đã lưu từ NB3 và phân tích kỹ lưỡng kết quả.

---

### Cell-by-cell giải thích

#### 📦 Cell 1 — Import & Config
- Cùng thư viện với NB3 nhưng không có training libs
- `plt.rcParams` → chuẩn hóa style đồ thị

#### 📂 Cell 2 — Load Models & Test Data
```python
lr_model   = joblib.load('lr_model.pkl')
xgb_model  = joblib.load('xgb_model.pkl')
lgbm_model = joblib.load('best_model.pkl')
opt_thresh = joblib.load('optimal_threshold.pkl')  # 0.5971
test_data  = joblib.load('test_data.pkl')
X_test, X_test_lr, y_test = ...
```
- Load 3 models đã train + test set (12,346 dòng)
- **Không có train data ở đây** — NB4 chỉ thấy test set

#### 🔢 Cell 3 — Compute Predictions
```python
model_configs = [
    (lr_model,   X_test_lr, 0.5,       'Logistic Regression'),
    (xgb_model,  X_test,    0.5,       'XGBoost'),
    (lgbm_model, X_test,    0.5971,    'LightGBM'),
]
for model, X, thresh, name in model_configs:
    probs = model.predict_proba(X)[:, 1]   # xác suất tái mua
    preds = (probs >= thresh).astype(int)  # 0 hoặc 1
    # tính 5 metrics: acc, prec, rec, f1, auc
```
- **LR** dùng `X_test_lr` (scaled) — vì LR cần normalized input
- **XGBoost / LightGBM** dùng `X_test` (unscaled) — tree model không cần
- **LightGBM** dùng threshold 0.5971 (từ PR curve) — khác 0.5 mặc định

#### 📊 Cells còn lại (Section 1-7)
| Section | Mô tả |
|---------|-------|
| 1 | Bảng so sánh có màu (best=xanh, worst=đỏ) |
| 2 | ROC Curve — 3 models (AUC) |
| 3 | Precision-Recall Curve (AP) |
| 4 | Confusion Matrix — 3 subplots |
| 5 | Feature Importance bar chart (LGBM vs XGB) |
| 6 | Error Analysis theo product_category |
| 7 | Kết luận & đề xuất |

---

## 💡 TÓM TẮT CÁC KHÁI NIỆM QUAN TRỌNG

### Tại sao cần 2 notebook riêng biệt?
- **NB3 (Training)** = quá trình "học" của model — chỉ được nhìn train data
- **NB4 (Evaluation)** = quá trình "thi" của model — chỉ dùng test data
- Tách biệt để đảm bảo kết quả evaluation là **khách quan, không bị ảnh hưởng bởi quá trình học**

### Data Leakage là gì?
- Khi thông tin từ test set (hoặc nhãn) vô tình "rò rỉ" vào training
- Hậu quả: model đạt F1 ảo (0.99) nhưng thực tế vô dụng
- Trong project này: `total_interactions` chứa `purchase_count` → biết luôn nhãn

### Threshold (Ngưỡng phân loại) là gì?
- Model trả về **xác suất** (0 → 1), không phải nhãn trực tiếp
- `threshold = 0.5`: nếu xác suất ≥ 0.5 → dự đoán "tái mua"
- Điều chỉnh threshold để cân bằng Precision/Recall theo mục tiêu kinh doanh
