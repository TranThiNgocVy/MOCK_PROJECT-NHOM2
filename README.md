# 🛒 Repurchase Prediction — ML Pipeline

Dự đoán xác suất khách hàng **tái mua trong cùng product_category** dựa trên dữ liệu hành vi e-commerce.  
Bài toán: **Binary Classification** · Target: `repurchase` (0 = không tái mua, 1 = tái mua)

---

## 📁 Cấu trúc thư mục

```
mini_project_ml/
│
├── 1_Data_Generation.ipynb         # Load & khám phá nguồn dữ liệu
├── 2_Exploratory_Data_Analysis.ipynb  # EDA đầy đủ
├── 3_Model_Building.ipynb          # Feature engineering + train 3 models
├── 4_Model_Evaluation.ipynb        # So sánh & đánh giá toàn diện
├── README.md
│
├── data/
│   ├── synthetic_data.csv          # Bản copy từ nguồn gốc
│   ├── processed_data.csv          # Sau preprocessing pipeline
│   ├── derived_features.csv        # Các biến phái sinh
│   └── feature_data.csv            # Dataset cuối (gốc + derived)
│
└── models/
    ├── best_model.pkl              # LightGBM tốt nhất
    ├── feature_importance.csv      # Tầm quan trọng đặc trưng
    ├── optimal_threshold.pkl       # Ngưỡng tối ưu F1
    └── preprocessor.pkl            # sklearn Pipeline
```

---

## ⚙️ Setup môi trường

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
joblib>=1.2.0
imbalanced-learn>=0.10.0
```

---

## 🚀 Hướng dẫn chạy

Chạy theo thứ tự từ Notebook 1 → 4:

| Bước | Notebook | Mô tả |
|:----:|----------|-------|
| 1 | `1_Data_Generation.ipynb` | Load data gốc, tìm hiểu schema, lưu bản copy |
| 2 | `2_Exploratory_Data_Analysis.ipynb` | Phân tích phân phối, correlation, outlier |
| 3 | `3_Model_Building.ipynb` | Preprocessing + Feature Engineering + Train 3 models |
| 4 | `4_Model_Evaluation.ipynb` | So sánh models, chọn model tốt nhất |

---

## 🤖 3 Models được huấn luyện

| Model | Vai trò | Đặc điểm |
|-------|---------|-----------|
| **Logistic Regression** | Baseline | Linear boundary, giải thích được, nhanh |
| **XGBoost** | Nâng cao | Gradient boosting, bắt non-linear patterns |
| **LightGBM** ⭐ | Best model | Tốc độ cao, leaf-wise, optimal threshold |

**Primary metric:** F1-Score (vì class imbalance ~40/60)  
**Secondary:** ROC-AUC, Precision, Recall

---

## 📊 Dữ liệu

- **Nguồn:** `data_repurchase.csv` (~61,728 dòng × 21 cột)
- **Key:** `(user_id, product_category)`
- **Target:** `repurchase` — gán nhãn bằng IC-Weighted behavioral score + Otsu threshold
- **Lưu ý:** `cart_rate` có thể > 1 (direct-to-cart, hợp lệ); `avg_rating` có ~28% NaN

---

## 🛠️ Tech Stack

Python · scikit-learn · XGBoost · LightGBM · pandas · NumPy · seaborn · matplotlib · scipy · joblib
