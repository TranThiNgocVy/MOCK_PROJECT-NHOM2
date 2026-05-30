# 📚 BỘ CÂU HỎI PHẢN BIỆN — MOCK PROJECT ML: REPURCHASE PREDICTION

> **Đề tài:** Dự đoán tái mua hàng trong cùng product_category (Binary Classification)
> **Dataset:** 61,728 cặp (user × category) × 20 cột
> **Models:** Logistic Regression, XGBoost, LightGBM
> **KPI:** F1 ≥ 0.72 | ROC-AUC ≥ 0.87 | Recall ≥ 0.68
> **Best model:** LightGBM (θ* = 0.5041)

---

## 📋 FORMAT MỖI CÂU HỎI

- **Câu hỏi:** Nội dung câu hỏi
- **Mục đích:** Giảng viên muốn kiểm tra gì
- **Trả lời chuẩn:** Câu trả lời đầy đủ, chính xác
- **Thường sai ở:** Lỗi phổ biến khi trả lời
- **Trả lời nhanh:** 1–2 câu khi bị hỏi nhanh trên lớp
- **Mức độ:** Easy / Medium / Hard / Very Hard

---

## 1. PROBLEM & BUSINESS UNDERSTANDING

### Câu 1.1
- **Câu hỏi:** Bài toán này giải quyết vấn đề kinh doanh gì? Tại sao chọn binary classification?
- **Mục đích:** Kiểm tra sinh viên hiểu business context, không chỉ làm kỹ thuật thuần túy.
- **Trả lời chuẩn:** Nền tảng e-commerce Việt Nam muốn cá nhân hóa chiến dịch CRM. ~67% khách không quay lại category sau lần mua đầu → chi phí acquisition bị lãng phí. Bài toán dự đoán cặp (user, category): khách có tái mua trong category đó không? Output 0/1 → binary classification. Kết quả giúp CRM gửi voucher đúng người, giảm chi phí marketing.
- **Thường sai ở:** Nói "làm binary classification vì có 2 class" mà không giải thích business context. Không đề cập unit of analysis là cặp (user, category).
- **Trả lời nhanh:** Dự đoán khách có tái mua trong cùng category không → giúp CRM target đúng người, tránh lãng phí voucher cho ~67% khách ít có khả năng tái mua.
- **Mức độ:** Easy

---

### Câu 1.2
- **Câu hỏi:** Tại sao unit of analysis là cặp (user, category) chứ không phải từng user hay từng đơn hàng?
- **Mục đích:** Kiểm tra hiểu biết về data granularity và business logic.
- **Trả lời chuẩn:** Một user có thể mua nhiều category với loyalty khác nhau. Ví dụ: khách trung thành với "áo thun" nhưng chỉ mua "đồ điện" 1 lần. Aggregate theo (user, category) giúp model học category-specific loyalty — phù hợp gửi voucher theo category. Nếu aggregate theo toàn user → mất thông tin category preference.
- **Thường sai ở:** Không giải thích được tại sao không dùng per-transaction hoặc per-user.
- **Trả lời nhanh:** Mỗi khách có loyalty khác nhau với từng category. Aggregate theo (user, category) giúp model học được category loyalty riêng biệt.
- **Mức độ:** Medium

---

### Câu 1.3
- **Câu hỏi:** KPI mục tiêu F1 ≥ 0.72, ROC-AUC ≥ 0.87, Recall ≥ 0.68 được đặt ra dựa trên cơ sở nào?
- **Mục đích:** Kiểm tra KPI có logic hay đặt bừa.
- **Trả lời chuẩn:** F1 ≥ 0.72 dựa trên imbalance 67/33 — model phải vượt baseline ngẫu nhiên (~0.5). ROC-AUC ≥ 0.87 là standard benchmark cho bài toán hành vi khách hàng. Recall ≥ 0.68 dựa trên cost analysis: FN cost (150,000 VND/người bỏ lỡ) = 5× FP cost (30,000 VND/voucher lãng phí) → cần recall cao hơn precision.
- **Thường sai ở:** Không giải thích cơ sở từng KPI, đặt KPI cảm tính.
- **Trả lời nhanh:** FN đắt gấp 5× FP (150K vs 30K VND) → cần Recall cao. F1 ≥ 0.72 vượt baseline imbalance. ROC-AUC ≥ 0.87 = benchmark industry.
- **Mức độ:** Medium

---

### Câu 1.4
- **Câu hỏi:** Tại sao primary metric là F1 mà không phải Accuracy?
- **Mục đích:** Câu hỏi cơ bản nhưng quan trọng về metric selection cho imbalanced data.
- **Trả lời chuẩn:** Class distribution 67/33 — model luôn predict class 0 → Accuracy = 67.2% (không học gì). F1 = harmonic mean(Precision, Recall) → penalize mạnh khi bỏ sót minority class. Model "lazy" predict all 0: Recall class 1 = 0 → F1 = 0. F1 phù hợp hơn khi cost FP ≠ FN (150K vs 30K VND).
- **Thường sai ở:** Nói "accuracy thấp vì dataset nhỏ" — sai, vấn đề là imbalance không phải kích thước.
- **Trả lời nhanh:** Accuracy bị mislead bởi majority class (67% baseline). F1 cân bằng Precision/Recall → phạt mạnh khi bỏ sót minority.
- **Mức độ:** Easy

---

## 2. SYNTHETIC DATA GENERATION

### Câu 2.1
- **Câu hỏi:** Tại sao dữ liệu gọi là "synthetic"? Dataset này có phải dữ liệu thực không?
- **Mục đích:** Kiểm tra tính trung thực về nguồn dữ liệu.
- **Trả lời chuẩn:** Đây là dữ liệu tổng hợp mô phỏng hành vi e-commerce Việt Nam. Các đặc trưng được thiết kế để phản ánh pattern thực tế (phân phối purchase_count, ~28% missing avg_rating, cart_rate > 1). Không phải dữ liệu từ hệ thống thực. Trong production cần validate lại với real data.
- **Thường sai ở:** Nói "dữ liệu thực từ e-commerce" khi không có bằng chứng.
- **Trả lời nhanh:** Dữ liệu tổng hợp mô phỏng e-commerce, có các đặc điểm thực tế như imbalance và missing values. Không phải real production data.
- **Mức độ:** Easy

---

### Câu 2.2
- **Câu hỏi:** Nhãn `repurchase` được tạo như thế nào? Tại sao dùng Otsu thay vì đặt ngưỡng tay hoặc dùng Q2/median?
- **Mục đích:** Câu hỏi core — kiểm tra hiểu biết về label generation.
- **Trả lời chuẩn:** Nhãn tạo từ `purchase_count` (số lần mua trong cùng category) qua thuật toán Otsu 1D. Otsu tìm T* = argmax ω₀·ω₁·(μ₀−μ₁)² bằng cách quét 500 ngưỡng → T* = 4.02. Lý do dùng Otsu thay vì Q2: (1) Không bias — T* xác định bởi phân phối thực, (2) Tối đa hóa inter-class variance → 2 nhóm phân tách rõ nhất, (3) Khách quan, reproducible. Q2 = ngưỡng cứng 50/50 không tính đến hình dạng phân phối thực tế.
- **Thường sai ở:** Giải thích sai công thức Otsu, không phân biệt Otsu vs Q2, không giải thích T*=4.02 → purchase_count ≥ 5 (số nguyên đầu tiên vượt ngưỡng).
- **Trả lời nhanh:** Otsu tối đa hóa khoảng cách 2 nhóm theo phân phối thực → T*=4.02, khách quan hơn đặt ngưỡng tay hoặc median cứng.
- **Mức độ:** Hard

---

### Câu 2.3
- **Câu hỏi:** Giải thích công thức Otsu: T* = argmax ω₀·ω₁·(μ₀−μ₁)². Tại sao maximize inter-class variance?
- **Mục đích:** Kiểm tra hiểu biết toán học của Otsu.
- **Trả lời chuẩn:** ω₀, ω₁ là tỷ lệ 2 nhóm tại threshold T. μ₀, μ₁ là trung bình purchase_count của 2 nhóm. (μ₀−μ₁)² là khoảng cách 2 trung bình. ω₀·ω₁ penalize khi 1 nhóm quá nhỏ. Maximize inter-class variance = 2 nhóm cách xa nhau nhất + kích thước cân đối → phân tách rõ nhất. Đây là Fisher's criterion, cũng là nền tảng Linear Discriminant Analysis.
- **Thường sai ở:** Không giải thích được ý nghĩa từng term trong công thức.
- **Trả lời nhanh:** Maximize khoảng cách 2 nhóm (μ₀−μ₁)² trong khi cân bằng kích thước (ω₀·ω₁) → tìm ngưỡng phân tách tốt nhất.
- **Mức độ:** Hard

---

### Câu 2.4
- **Câu hỏi:** T* = 4.02 là số thực, nhưng purchase_count là số nguyên. Kết quả cuối là gì?
- **Mục đích:** Kiểm tra hiểu biết về floating-point threshold trong Otsu.
- **Trả lời chuẩn:** Otsu quét midpoints của histogram bins — không nhất thiết là số nguyên. T*=4.02 → purchase_count ≥ 4.02 → thực tế là purchase_count ≥ 5 (số nguyên đầu tiên vượt ngưỡng). Class distribution kết quả: 32.8% tái mua (≥5 lần), 67.2% không tái mua (<5 lần).
- **Thường sai ở:** Nói "phải làm tròn lên 5" mà không giải thích cơ chế Otsu dùng histogram bins liên tục.
- **Trả lời nhanh:** Otsu dùng histogram bins nên T* là số thực. purchase_count integer nên ≥4.02 tương đương ≥5.
- **Mức độ:** Medium

---

## 3. DATA UNDERSTANDING & EDA

### Câu 3.1
- **Câu hỏi:** `cart_rate` có thể lớn hơn 1 không? Đây là lỗi dữ liệu hay hợp lệ?
- **Mục đích:** Kiểm tra hiểu biết về domain và data semantics.
- **Trả lời chuẩn:** cart_rate = total_cart / total_click. Giá trị > 1 là HOÀN TOÀN HỢP LỆ vì trong e-commerce, khách có thể thêm giỏ hàng trực tiếp (direct-to-cart) mà không click qua trang sản phẩm, hoặc 1 lần click dẫn đến nhiều lần add-to-cart. Không clip hay loại bỏ giá trị này vì đây là tín hiệu hành vi thực.
- **Thường sai ở:** Nói "cart_rate > 1 là lỗi dữ liệu, cần clip về [0,1]" — đây là sai nghiêm trọng về domain understanding.
- **Trả lời nhanh:** Hợp lệ — direct-to-cart hoặc 1 click → nhiều lần add-to-cart. Không clip, giữ nguyên.
- **Mức độ:** Medium

---

### Câu 3.2
- **Câu hỏi:** avg_rating có ~28% NaN. Chiến lược xử lý là gì? Tại sao?
- **Mục đích:** Kiểm tra khả năng phân tích missing data và chọn strategy phù hợp.
- **Trả lời chuẩn:** Strategy: impute NaN bằng global median (4.0). Lý do: (1) 28% cao — không thể drop rows, (2) Median bền với outlier hơn mean, (3) Phân tích cross-tab cho thấy tỷ lệ tái mua của nhóm NaN và non-NaN tương đương → MCAR → impute hợp lý. Lưu ý: best practice là fit imputer trên X_train only rồi transform val/test — trong project impute pre-split là limitation đã được identify.
- **Thường sai ở:** Không biết phân biệt MCAR/MAR/MNAR, không đề cập vấn đề impute trước hay sau split.
- **Trả lời nhanh:** Impute median 4.0 vì 28% không thể drop, median bền với outlier, pattern NaN không liên quan target (MCAR). Limitation: nên fit imputer trên train only.
- **Mức độ:** Medium

---

### Câu 3.3
- **Câu hỏi:** Tại sao `dominant_gender` và `dominant_location` bị drop khỏi feature matrix?
- **Mục đích:** Kiểm tra kết quả EDA và khả năng đưa ra quyết định data-driven.
- **Trả lời chuẩn:** Chi-Square test cho thấy p-value > 0.05 với cả 2 biến → không có quan hệ thống kê với target. Cramér's V ≈ 0 → không có practical significance. Kết luận: giới tính và địa điểm không ảnh hưởng đến hành vi tái mua trong category. Drop để giảm dimensionality và tránh nhiễu.
- **Thường sai ở:** Drop biến mà không giải thích được test thống kê, không biết Cramér's V là gì.
- **Trả lời nhanh:** Chi-square p > 0.05 và Cramér's V ≈ 0 → không có quan hệ thống kê với target → drop để giảm noise.
- **Mức độ:** Medium

---

### Câu 3.4
- **Câu hỏi:** Tại sao `total_interactions` bị xóa khỏi dataset?
- **Mục đích:** Câu hỏi bẫy — kiểm tra hiểu biết về indirect data leakage.
- **Trả lời chuẩn:** `total_interactions = total_view + total_click + total_cart + total_wishlist + purchase_count`. Vì purchase_count đã được dùng để tạo nhãn repurchase, nếu giữ total_interactions → model suy ra purchase_count = total_interactions − view − click − cart − wishlist → 100% accuracy với 1 phép tính → DATA LEAKAGE gián tiếp. Phải xóa total_interactions ngay khi xóa purchase_count.
- **Thường sai ở:** Không nhận ra total_interactions chứa purchase_count, không phát hiện leakage gián tiếp này.
- **Trả lời nhanh:** total_interactions = view+click+cart+wishlist+purchase_count → chứa purchase_count (đã làm nhãn) → leakage gián tiếp → phải xóa.
- **Mức độ:** Hard

---

## 4. DATA CLEANING & PREPROCESSING

### Câu 4.1
- **Câu hỏi:** Tại sao `purchase_count` phải bị xóa khỏi feature matrix ngay sau khi tạo nhãn?
- **Mục đích:** Kiểm tra hiểu biết về data leakage — câu hỏi cực kỳ quan trọng.
- **Trả lời chuẩn:** purchase_count là biến DUY NHẤT dùng để tạo nhãn repurchase (Otsu T*=4.02). Nếu giữ: model học purchase_count ≥ 5 → predict 1 → accuracy 100% nhưng vô nghĩa (model học nhãn, không học pattern thực). Trong production, purchase_count tương lai không biết trước → model không generalize. Đây là target leakage nghiêm trọng nhất.
- **Thường sai ở:** Giải thích sơ sài, không nói được hậu quả cụ thể (100% accuracy giả), không đề cập production scenario.
- **Trả lời nhanh:** Giữ purchase_count = model học chính nhãn, accuracy 100% giả. Thực tế không biết purchase_count tương lai → phải xóa để model học hành vi gián tiếp thực sự.
- **Mức độ:** Hard

---

### Câu 4.2
- **Câu hỏi:** Tại sao Logistic Regression cần StandardScaler nhưng XGBoost và LightGBM thì không?
- **Mục đích:** Kiểm tra hiểu biết về bản chất các thuật toán.
- **Trả lời chuẩn:** LR tối ưu bằng gradient descent — nếu features có scale khác nhau (avg_price hàng triệu vs click_through_rate trong [0,1]) → gradient descent hội tụ chậm, coefficients không so sánh được. StandardScaler chuẩn hóa mean=0, std=1. Tree-based models (XGB, LGBM) chỉ dùng thứ tự giá trị để split → scale không ảnh hưởng → không cần scale.
- **Thường sai ở:** Nói "scale để model tốt hơn" mà không giải thích cơ chế, hoặc nói "tree models cũng cần scale để so sánh feature importance".
- **Trả lời nhanh:** LR dùng gradient descent → nhạy với scale. Tree models dùng thứ tự để split → scale không ảnh hưởng.
- **Mức độ:** Medium

---

### Câu 4.3
- **Câu hỏi:** Tại sao StandardScaler chỉ được `fit` trên X_train và `transform` trên cả train/val/test?
- **Mục đích:** Kiểm tra kiến thức về preprocessing leakage.
- **Trả lời chuẩn:** Nếu fit scaler trên toàn bộ dataset (train+val+test): mean và std tính từ val/test data → scaler "biết trước" distribution của val/test → leakage. Trong production, dữ liệu mới không có trong train → phải dùng statistics từ train. Fit trên train only → transform val/test = đúng production pipeline.
- **Thường sai ở:** Fit scaler trên toàn dataset trước khi split — lỗi rất phổ biến.
- **Trả lời nhanh:** Fit trên val/test = model biết distribution của val/test trước → leakage. Fit chỉ trên train → đúng production pipeline.
- **Mức độ:** Medium

---

## 5. STATISTICAL ANALYSIS

### Câu 5.1
- **Câu hỏi:** Point-Biserial correlation khác gì Pearson correlation? Khi nào dùng?
- **Mục đích:** Kiểm tra kiến thức thống kê về correlation với biến binary.
- **Trả lời chuẩn:** Point-Biserial là trường hợp đặc biệt của Pearson khi một biến là binary (0/1). Công thức và kết quả hoàn toàn tương đương Pearson, nhưng nhấn mạnh về conceptual — so sánh mean của continuous variable giữa 2 nhóm binary. Dùng Point-Biserial khi đo correlation giữa continuous feature và binary target.
- **Thường sai ở:** Nói "Point-Biserial hoàn toàn khác Pearson" — thực ra là special case.
- **Trả lời nhanh:** Point-Biserial = Pearson đặc biệt khi 1 biến là binary. Kết quả tương đương, chỉ khác cách frame.
- **Mức độ:** Medium

---

### Câu 5.2
- **Câu hỏi:** Cramér's V đo gì? Ngưỡng V bao nhiêu là có ý nghĩa?
- **Mục đích:** Kiểm tra kiến thức về categorical association.
- **Trả lời chuẩn:** Cramér's V đo độ mạnh của liên kết giữa 2 categorical variables, range [0,1]. V=0 → không liên kết, V=1 → liên kết hoàn toàn. Tính từ Chi-square: V = sqrt(χ²/(n × min_dim)). Ngưỡng tham chiếu: V < 0.1 = yếu, V ∈ [0.1, 0.3] = vừa, V > 0.3 = mạnh. Trong project: dominant_gender và dominant_location có V ≈ 0 → drop.
- **Thường sai ở:** Nhầm Cramér's V với Pearson correlation, không biết range và interpretation.
- **Trả lời nhanh:** V đo association categorical-categorical, range [0,1]. V < 0.1 = weak. dominant_gender/location V≈0 → drop là hợp lý.
- **Mức độ:** Medium

---

## 6. FEATURE ENGINEERING

### Câu 6.1
- **Câu hỏi:** Tại sao thêm +1 vào mẫu số trong các derived features (ví dụ: view_to_click_ratio = total_click/(total_view+1))?
- **Mục đích:** Kiểm tra hiểu biết về Laplace smoothing.
- **Trả lời chuẩn:** +1 vào mẫu số (Laplace smoothing / add-one smoothing) giải quyết 2 vấn đề: (1) Division by zero — total_view = 0 → ZeroDivisionError, (2) Extreme values — 1 click/1 view → rate = 1.0 nhưng 0 click/0 view → undefined. +1 đảm bảo mẫu số ≥ 1, giá trị feature hợp lý. Kỹ thuật chuẩn trong NLP (Laplace smoothing cho n-gram) áp dụng vào feature engineering.
- **Thường sai ở:** Chỉ nói "+1 để tránh chia cho 0" mà không đề cập Laplace smoothing và extreme value handling.
- **Trả lời nhanh:** Laplace smoothing: tránh chia cho 0 + hạn chế extreme values khi denominator nhỏ. Kỹ thuật chuẩn trong feature engineering.
- **Mức độ:** Medium

---

### Câu 6.2
- **Câu hỏi:** `pure_beh = total_view + total_click + total_cart + total_wishlist`. Tại sao không bao gồm purchase_count?
- **Mục đích:** Kiểm tra anti-leakage awareness trong feature engineering.
- **Trả lời chuẩn:** purchase_count đã bị xóa khỏi dataset vì nó là source của nhãn repurchase (Otsu T*=4.02) → data leakage. pure_beh đại diện cho "hành vi tương tác thuần" — không bao gồm hành vi mua hàng trực tiếp. Intentional design: model phải dự đoán từ tín hiệu gián tiếp (xem, click, giỏ hàng, wishlist) không phải từ số lần đã mua.
- **Thường sai ở:** Không nhận ra connection giữa exclude purchase_count khỏi pure_beh với anti-leakage.
- **Trả lời nhanh:** purchase_count đã bị xóa (anti-leakage), pure_beh chỉ dùng hành vi tương tác gián tiếp.
- **Mức độ:** Hard

---

### Câu 6.3
- **Câu hỏi:** Giải thích ý nghĩa business của `discount_sensitivity` và `brand_loyalty_score`.
- **Mục đích:** Kiểm tra khả năng interpret features theo business context.
- **Trả lời chuẩn:** discount_sensitivity = avg_discount/(avg_price+1e-6): khách có avg_discount cao so với avg_price → nhạy cảm với giá, mua chủ yếu khi có KM → khả năng tái mua thấp khi không có KM. brand_loyalty_score = total_cart/(unique_brands+1): cao khi ít brand nhưng nhiều lần add-to-cart → loyal với 1-2 brand → có thể tái mua cao.
- **Thường sai ở:** Chỉ giải thích công thức toán học mà không nói được business meaning.
- **Trả lời nhanh:** discount_sensitivity đo phụ thuộc khuyến mãi, brand_loyalty_score đo tập trung vào ít brand — cả hai là signals về hành vi mua trung thành.
- **Mức độ:** Medium

---

### Câu 6.4
- **Câu hỏi:** Tại sao tạo 15 derived features? Có bị overfitting do quá nhiều features không?
- **Mục đích:** Kiểm tra critical thinking về feature explosion.
- **Trả lời chuẩn:** 15 derived features từ 14 raw features (ratio/interaction terms) — mỗi feature có business meaning rõ ràng. Tổng 29 features không quá lớn với 61K samples. Feature Selection (Bước 6) sau đó dùng correlation + mutual info + VIF để loại features kém. Tree-based models tự handle feature selection qua splitting → ít rủi ro overfitting hơn LR.
- **Thường sai ở:** Không đề cập Feature Selection step sau đó, không giải thích business justification.
- **Trả lời nhanh:** 15 features có business meaning rõ ràng, Feature Selection sau đó loại features kém → không phải random feature explosion.
- **Mức độ:** Medium

---

## 7. FEATURE SELECTION

### Câu 7.1
- **Câu hỏi:** Tại sao dùng kết hợp Correlation + Mutual Information + VIF thay vì chỉ 1 phương pháp?
- **Mục đích:** Kiểm tra hiểu biết về complementary nature của các phương pháp.
- **Trả lời chuẩn:** Correlation (Pearson/Point-Biserial) chỉ đo linear relationship. Mutual Information đo cả non-linear relationship. VIF đo redundancy giữa features. Kết hợp 3: feature có correlation thấp nhưng MI cao → non-linear signal → nên giữ. Feature VIF cao → collinear → xem xét drop. 3 phương pháp complementary → selection robust hơn.
- **Thường sai ở:** Không giải thích được limitation của từng phương pháp riêng lẻ.
- **Trả lời nhanh:** Correlation bắt linear, MI bắt non-linear, VIF bắt redundancy — dùng cả 3 để không bỏ sót và không giữ features thừa.
- **Mức độ:** Hard

---

### Câu 7.2
- **Câu hỏi:** Multicollinearity là gì? VIF ngưỡng bao nhiêu thì cần xử lý?
- **Mục đích:** Kiểm tra kiến thức về multicollinearity.
- **Trả lời chuẩn:** Multicollinearity = 2+ features tương quan cao với nhau. Ảnh hưởng: LR coefficients bất ổn, khó interpret, standard error lớn. Tree models: ít ảnh hưởng accuracy nhưng feature importance bị dilute. VIF (Variance Inflation Factor) > 10 → multicollinearity cao → nên drop 1 trong 2 features. VIF = 1/(1-R²) của feature đó khi regress lên các features còn lại.
- **Thường sai ở:** Không biết VIF, nói multicollinearity không ảnh hưởng tree models (đúng với performance nhưng ảnh hưởng interpretation).
- **Trả lời nhanh:** VIF > 10 → multicollinearity cao → drop 1 trong 2 correlated features. Tree models ít bị ảnh hưởng accuracy nhưng feature importance bị dilute.
- **Mức độ:** Hard

---

## 8. TRAIN/TEST SPLIT & LEAKAGE PREVENTION

### Câu 8.1
- **Câu hỏi:** Tại sao dùng split 3 tập 70/15/15 thay vì 80/20?
- **Mục đích:** Kiểm tra hiểu biết về validation set role.
- **Trả lời chuẩn:** Validation set (15%) dùng cho: (1) Hyperparameter tuning và early stopping trong XGB/LGBM, (2) Threshold optimization — tìm optimal threshold trên val, không trên test, (3) Model selection. Test set TUYỆT ĐỐI chỉ dùng 1 lần report final performance. Tune threshold trên test = data snooping → overly optimistic results.
- **Thường sai ở:** Không giải thích được tại sao cần val riêng, nhầm lẫn val và test purposes.
- **Trả lời nhanh:** Val set dành cho hyperparameter tuning và threshold optimization. Test set chỉ 1 lần để report. Tune trên test → data snooping → optimistic bias.
- **Mức độ:** Medium

---

### Câu 8.2
- **Câu hỏi:** Có nên dùng stratified split không? Tại sao?
- **Mục đích:** Kiểm tra awareness về class imbalance trong splitting.
- **Trả lời chuẩn:** Có, stratified split đảm bảo tỷ lệ class (67/33) được duy trì trong cả 3 tập. Không dùng stratified: ngẫu nhiên có thể train có 70% class 0, val có 80% → distribution shift → evaluation không ổn định. Với 61K samples nguy cơ nhỏ hơn nhưng vẫn nên dùng stratified như best practice. sklearn train_test_split với stratify=y tự xử lý.
- **Thường sai ở:** Không đề cập stratified split, chỉ nói random split.
- **Trả lời nhanh:** Stratified split giữ tỷ lệ class trong cả 3 tập → evaluation ổn định, không bị distribution shift.
- **Mức độ:** Medium

---

## 9. MODELING

### Câu 9.1
- **Câu hỏi:** Tại sao chọn Logistic Regression, XGBoost, và LightGBM?
- **Mục đích:** Kiểm tra khả năng justify model choices.
- **Trả lời chuẩn:** LR = baseline interpretable, fast, establish lower bound. XGBoost = gradient boosting mạnh, tabular data SOTA, mature ecosystem. LightGBM = nhanh hơn XGBoost (histogram-based), leaf-wise growth → thường tốt hơn tabular. 3 models tạo diversity: linear (LR) vs non-linear ensemble (XGB, LGBM). Benchmark Kaggle/AutoML: LGBM thường top performer trên tabular data với 50K+ samples.
- **Thường sai ở:** Không giải thích LR là baseline, không biết LightGBM leaf-wise growth.
- **Trả lời nhanh:** LR = baseline. XGBoost = proven SOTA tabular. LightGBM = faster + better tabular nhờ leaf-wise. Diversity linear vs non-linear.
- **Mức độ:** Medium

---

### Câu 9.2
- **Câu hỏi:** `scale_pos_weight` trong XGBoost dùng để làm gì? Tính như thế nào?
- **Mục đích:** Kiểm tra hiểu biết về imbalanced handling trong XGBoost.
- **Trả lời chuẩn:** scale_pos_weight = n_negative / n_positive = 41,507 / 20,221 ≈ 2.05. Giá trị này tăng weight của minority class (class 1) lên 2.05× → model penalize nhiều hơn khi miss class 1 → tăng recall cho minority class. Tương đương class_weight='balanced' trong LR và LightGBM (API khác, concept tương đương).
- **Thường sai ở:** Không biết công thức tính, nhầm là "weight của từng sample".
- **Trả lời nhanh:** scale_pos_weight = neg_count/pos_count = 41507/20221 ≈ 2.05 → tăng penalty khi miss class 1 → recall minority tăng.
- **Mức độ:** Medium

---

### Câu 9.3
- **Câu hỏi:** LightGBM leaf-wise growth khác gì XGBoost level-wise growth? Ưu/nhược điểm?
- **Mục đích:** Kiểm tra kiến thức kỹ thuật về tree growing strategy.
- **Trả lời chuẩn:** Level-wise (XGBoost): mỗi iteration mở rộng tất cả nodes cùng depth → balanced tree, ít overfit. Leaf-wise (LightGBM): chọn leaf có maximum loss reduction để split tiếp → asymmetric trees, đạt loss thấp hơn với cùng số splits → nhanh hơn và accuracy cao hơn trên tabular. Nhược điểm leaf-wise: dễ overfit hơn với dataset nhỏ — cần min_data_in_leaf. Với 61K samples → LightGBM safe.
- **Thường sai ở:** Mô tả ngược (LGBM = level-wise), không biết trade-off overfitting.
- **Trả lời nhanh:** Level-wise (XGB) = balanced tree, ít overfit. Leaf-wise (LGBM) = chọn leaf tốt nhất → accuracy cao hơn, nhanh hơn nhưng cần regularize với data nhỏ.
- **Mức độ:** Hard

---

## 10. HYPERPARAMETER TUNING

### Câu 10.1
- **Câu hỏi:** `early_stopping_rounds=50` có nghĩa là gì? Tại sao cần early stopping?
- **Mục đích:** Kiểm tra hiểu biết về regularization qua early stopping.
- **Trả lời chuẩn:** early_stopping_rounds=50: nếu sau 50 rounds liên tiếp validation metric không cải thiện → dừng training. Mục đích: (1) Chống overfitting — train tiếp → fit noise, (2) Tiết kiệm computation — không cần train hết 500 rounds, (3) Automatic model selection — chọn model tốt nhất trên val. Monitor trên X_val, không phải X_test.
- **Thường sai ở:** Nói early stopping dùng test set để monitor, không giải thích tại sao 50 rounds.
- **Trả lời nhanh:** 50 rounds val metric không tăng → dừng → tránh overfit + tiết kiệm compute. Model chọn tại round best trên val.
- **Mức độ:** Medium

---

### Câu 10.2
- **Câu hỏi:** Hyperparameters n_estimators=500, learning_rate=0.05, max_depth=6 có được GridSearch không?
- **Mục đích:** Kiểm tra awareness về hyperparameter tuning process.
- **Trả lời chuẩn:** Trong project, hyperparameters set theo best practices từ literature cho tabular data: learning_rate=0.05 (standard boosting), max_depth=6 (standard XGBoost), num_leaves=63 (LGBM ≈ 2^6−1). Early stopping tự chọn n_estimators tốt nhất. GridSearch/RandomSearch/Optuna có thể cải thiện thêm nhưng tăng computation. Trade-off giữa optimization và practical constraints.
- **Thường sai ở:** Không biết values set theo best practices, không đề cập early stopping là implicit tuning.
- **Trả lời nhanh:** Set theo best practices từ literature, không GridSearch. Early stopping tự chọn n_estimators tối ưu. Optuna có thể improve thêm — improvement point rõ ràng.
- **Mức độ:** Medium

---

## 11. OVERFITTING / UNDERFITTING

### Câu 11.1
- **Câu hỏi:** Làm sao biết model có bị overfit không? Dấu hiệu trong kết quả?
- **Mục đích:** Kiểm tra khả năng diagnose model behavior.
- **Trả lời chuẩn:** Overfit: train metric >> val/test metric (gap lớn). Underfit: cả train và test đều thấp. Kiểm tra: (1) Compare train vs val F1/AUC → gap nhỏ = good generalization, (2) Learning curve: val metric plateau/giảm khi tiếp tục train → early stopping là đúng, (3) class_weight='balanced' tránh model lazy predict majority class. Trong project: early stopping + regularization handle overfit cho boosting.
- **Thường sai ở:** Không biết cách đọc learning curve.
- **Trả lời nhanh:** Overfit = train F1 >> test F1. Kiểm tra qua learning curve và train/val gap. Early stopping + regularization trong project giúp control overfit.
- **Mức độ:** Medium

---

### Câu 11.2
- **Câu hỏi:** Bias-Variance trade-off là gì? Model nào trong project dễ bị high bias/variance?
- **Mục đích:** Kiểm tra kiến thức ML theory.
- **Trả lời chuẩn:** Bias = model quá đơn giản, underfits (wrong assumptions). Variance = model quá phức tạp, overfits (sensitive to noise). Trong project: LR có bias cao (linear assumption) nhưng variance thấp → stable nhưng có thể underfit non-linear patterns. LightGBM/XGBoost có bias thấp (flexible) nhưng variance cao → cần regularization. Boosting giảm bias iteratively bằng cách fit residuals.
- **Thường sai ở:** Giải thích sai chiều trade-off, không đặt vào context project.
- **Trả lời nhanh:** LR = high bias/low variance (underfit risk). LGBM/XGB = low bias/high variance (overfit risk). Boosting giảm bias, regularization giảm variance.
- **Mức độ:** Hard

---

## 12. EVALUATION METRICS

### Câu 12.1
- **Câu hỏi:** Precision và Recall là gì? Trade-off như thế nào?
- **Mục đích:** Kiểm tra kiến thức cơ bản nhưng cần giải thích đúng context.
- **Trả lời chuẩn:** Precision = TP/(TP+FP) = trong số predict=tái mua, bao nhiêu % thực sự tái mua → cao = ít lãng phí voucher. Recall = TP/(TP+FN) = trong số thực sự tái mua, bao nhiêu % được bắt → cao = ít bỏ lỡ khách tiềm năng. Trade-off: giảm threshold → Recall↑ Precision↓. FN cost cao → ưu tiên Recall.
- **Thường sai ở:** Định nghĩa ngược Precision/Recall, không giải thích trade-off.
- **Trả lời nhanh:** Precision = bao nhiêu predict positive là đúng. Recall = bao nhiêu positive thực được bắt. Threshold giảm → Recall↑ Precision↓. FN đắt hơn → cần Recall cao.
- **Mức độ:** Easy

---

### Câu 12.2
- **Câu hỏi:** Giải thích 4 ô confusion matrix và ý nghĩa business từng loại error.
- **Mục đích:** Kiểm tra khả năng connect kỹ thuật với business impact.
- **Trả lời chuẩn:** TP (predict 1, thực 1): gửi voucher đúng người tái mua → hiệu quả. TN (predict 0, thực 0): không gửi cho người không tái mua → tiết kiệm. FP (predict 1, thực 0): voucher lãng phí → 30,000 VND/người. FN (predict 0, thực 1): bỏ lỡ khách tái mua tiềm năng → mất 150,000 VND/người. FN cost = 5× FP cost → ưu tiên minimize FN → tăng Recall.
- **Thường sai ở:** Không kết nối với business cost, không nhớ FN=5×FP.
- **Trả lời nhanh:** FP = voucher lãng phí 30K. FN = bỏ lỡ khách tái mua 150K — đắt gấp 5×. Chiến lược minimize FN → tăng Recall.
- **Mức độ:** Easy

---

## 13. THRESHOLD OPTIMIZATION

### Câu 13.1
- **Câu hỏi:** Tại sao không dùng threshold mặc định 0.5? Optimal threshold được tìm như thế nào?
- **Mục đích:** Kiểm tra hiểu biết về threshold tuning — phân biệt senior vs junior.
- **Trả lời chuẩn:** Threshold 0.5 là default arbitrary — không tối ưu cho imbalanced data hay asymmetric costs. Với 67/33 và FN cost cao, optimal threshold thường < 0.5 (predict positive nhiều hơn → tăng recall). Cách tìm: vẽ Precision-Recall curve trên VAL set → với mỗi threshold tính F1 → chọn threshold maximize F1. Thực hiện trên VAL (không phải test) → tránh data snooping. Lưu vào .pkl để reproduce.
- **Thường sai ở:** Tune threshold trên test set, không biết tại sao 0.5 không tối ưu.
- **Trả lời nhanh:** 0.5 arbitrary với imbalanced data. Tune trên val bằng PR curve → maximize F1. Lưu threshold vào pkl để reproduce.
- **Mức độ:** Hard

---

### Câu 13.2
- **Câu hỏi:** Nếu tăng threshold thì Precision và Recall thay đổi như thế nào?
- **Mục đích:** Kiểm tra hiểu sâu về threshold mechanics.
- **Trả lời chuẩn:** Tăng threshold → model chỉ predict positive khi rất chắc chắn → Precision↑ (ít FP), Recall↓ (nhiều FN). Giảm threshold → predict positive nhiều hơn → Recall↑, Precision↓. Ví dụ: LightGBM θ*=0.5041 (KB1 stable) vs XGBoost θ*=0.3892 (KB2 growth, recall↑). KB2 threshold thấp hơn → recall cao hơn → phù hợp chiến dịch growth.
- **Thường sai ở:** Nói ngược (threshold tăng → Recall tăng).
- **Trả lời nhanh:** Threshold tăng → Precision↑ Recall↓. Threshold giảm → Recall↑ Precision↓. KB2 dùng threshold thấp hơn để bắt nhiều tái mua hơn.
- **Mức độ:** Easy

---

## 14. ROC-AUC / PR-AUC

### Câu 14.1
- **Câu hỏi:** ROC-AUC là gì? AUC = 0.87 có ý nghĩa gì?
- **Mục đích:** Kiểm tra kiến thức về ROC-AUC.
- **Trả lời chuẩn:** ROC curve plot TPR (Recall) vs FPR ở các threshold khác nhau. AUC = area under curve = xác suất model rank một positive sample cao hơn negative sample. AUC = 0.87 → 87% thời gian, model predict score cao hơn cho khách tái mua thực so với không tái mua. Range: 0.5 (random) → 1.0 (perfect). Threshold-independent ranking metric.
- **Thường sai ở:** Nói AUC là "accuracy at best threshold" — sai. AUC là ranking metric.
- **Trả lời nhanh:** AUC = P(score positive > score negative) = 0.87 → model rank đúng 87% các cặp. Threshold-independent.
- **Mức độ:** Medium

---

### Câu 14.2
- **Câu hỏi:** Khi nào PR-AUC tốt hơn ROC-AUC để đánh giá model?
- **Mục đích:** Kiểm tra kiến thức sâu về metric selection cho imbalanced data.
- **Trả lời chuẩn:** PR-AUC tốt hơn ROC-AUC cho imbalanced data. ROC dùng FPR = FP/(FP+TN) — với nhiều TN (majority class), FPR nhỏ ngay cả khi FP lớn → ROC bị inflate → AUC cao giả. PR dùng Precision (TP/(TP+FP)) không phụ thuộc TN → nhạy cảm hơn với minority class performance. Với class 67/33 → PR-AUC đáng tin hơn để so sánh models.
- **Thường sai ở:** Không biết sự khác biệt, nói ROC-AUC luôn tốt hơn.
- **Trả lời nhanh:** PR-AUC không bị inflate bởi nhiều TN → honest hơn với imbalanced data. ROC-AUC dễ mislead khi majority class chiếm đa số.
- **Mức độ:** Hard

---

## 15. IMBALANCED DATA HANDLING

### Câu 15.1
- **Câu hỏi:** Có những phương pháp nào xử lý imbalanced data? Project dùng phương pháp nào?
- **Mục đích:** Kiểm tra kiến thức về các chiến lược imbalance handling.
- **Trả lời chuẩn:** Phương pháp chính: (1) Resampling — SMOTE (oversample minority), Random Undersampling, (2) Cost-sensitive learning — class_weight='balanced', scale_pos_weight, (3) Threshold optimization, (4) Algorithm selection — Tree ensembles handle imbalance tốt hơn LR. Project dùng: class_weight='balanced' cho LR và LightGBM, scale_pos_weight cho XGBoost, kết hợp threshold optimization. Không dùng SMOTE vì: data đủ lớn (61K), ratio 67/33 không extreme, SMOTE tạo synthetic samples có thể introduce noise.
- **Thường sai ở:** Không biết scale_pos_weight = class_weight='balanced' tương đương, không biết tại sao không dùng SMOTE.
- **Trả lời nhanh:** Dùng class_weight='balanced' (LR/LGBM) và scale_pos_weight (XGB) — cost-sensitive. Không SMOTE vì ratio 67/33 không extreme và dataset đủ lớn.
- **Mức độ:** Medium

---

### Câu 15.2
- **Câu hỏi:** SMOTE là gì? Nếu dùng SMOTE thì áp dụng trên tập nào?
- **Mục đích:** Kiểm tra kiến thức về SMOTE và leakage trap.
- **Trả lời chuẩn:** SMOTE (Synthetic Minority Oversampling Technique) tạo synthetic samples cho minority class bằng cách interpolate giữa real minority samples và k nearest neighbors. QUAN TRỌNG: SMOTE chỉ được áp dụng trên TRAIN SET — không bao giờ trên val/test (leakage nếu áp dụng trên toàn dataset). Nhược điểm: synthetic samples có thể noisy, tăng training time. Với project này 67/33 không extreme → không cần.
- **Thường sai ở:** Apply SMOTE trên toàn dataset trước split — rất phổ biến và sai.
- **Trả lời nhanh:** SMOTE tạo synthetic minority samples. Chỉ apply trên TRAIN SET — không bao giờ trên val/test. Project không dùng vì ratio không extreme.
- **Mức độ:** Hard

---

## 16. EXPLAINABILITY & FEATURE IMPORTANCE

### Câu 16.1
- **Câu hỏi:** Feature importance của LightGBM tính như thế nào? Có limitations gì?
- **Mục đích:** Kiểm tra hiểu biết về built-in feature importance và giới hạn của nó.
- **Trả lời chuẩn:** Built-in feature importance thường dùng gain-based: tổng information gain khi feature được dùng để split. Hoặc split-count: số lần feature được chọn để split. Project dùng gain-based (thường reliable hơn). Limitations: (1) Inflate cho high-cardinality features, (2) Correlated features bị dilute importance giữa nhau, (3) Không phân biệt direction (positive/negative effect). SHAP cho interpretation tốt hơn — additive, consistent, address được các limitations trên.
- **Thường sai ở:** Nói feature importance là absolute truth, không biết limitations.
- **Trả lời nhanh:** Gain-based: tổng information gain khi split. Limitations: inflate high-cardinality, correlated features dilute nhau, không có direction. SHAP tốt hơn nhưng ngoài scope.
- **Mức độ:** Hard

---

### Câu 16.2
- **Câu hỏi:** Tại sao không dùng SHAP? Nếu thêm SHAP sẽ giải thích được gì hơn?
- **Mục đích:** Kiểm tra awareness về interpretability tools.
- **Trả lời chuẩn:** Project không dùng SHAP do scope constraint. SHAP nếu thêm vào sẽ cung cấp: (1) Local explanation — tại sao model predict khách X là tái mua, (2) Direction — feature nào PUSH prediction lên (+), feature nào kéo xuống (−), (3) Interaction effects, (4) Trustworthy global importance — không bị inflate hay dilute. SHAP waterfall plot cho từng prediction rất hữu ích khi present cho business stakeholders.
- **Thường sai ở:** Không biết SHAP là gì, hoặc nói "SHAP = feature importance" — SHAP là additive attribution.
- **Trả lời nhanh:** SHAP cho local explanation (tại sao predict khách X), direction (+/−), interaction. Built-in chỉ cho global ranking. Improvement point rõ ràng cho future work.
- **Mức độ:** Hard

---

## 17. PIPELINE & REPRODUCIBILITY

### Câu 17.1
- **Câu hỏi:** Project lưu những gì vào file `.pkl`? Tại sao cần lưu threshold?
- **Mục đích:** Kiểm tra hiểu biết về production-ready ML pipeline.
- **Trả lời chuẩn:** Project lưu: 3 models (best_model.pkl/LightGBM, xgb_model.pkl, lr_model.pkl), 3 thresholds (optimal_threshold.pkl/LGBM, xgb_optimal_threshold.pkl, lr_optimal_threshold.pkl), preprocessor.pkl (StandardScaler fitted trên train), test_data.pkl (X_test, X_test_lr, y_test, X_val, X_val_lr, y_val). Threshold phải lưu vì được tune trên val set — không lưu thì khi deploy phải tune lại hoặc dùng 0.5 suboptimal. Threshold là phần của inference pipeline.
- **Thường sai ở:** Không biết tại sao lưu threshold riêng, nói "threshold = 0.5 mặc định".
- **Trả lời nhanh:** Lưu model + threshold + preprocessor = full inference pipeline. Threshold tune trên val → phải lưu để reproduce khi deploy.
- **Mức độ:** Medium

---

### Câu 17.2
- **Câu hỏi:** Nếu deploy model lên production, cần chuẩn bị gì thêm ngoài model.pkl?
- **Mục đích:** Kiểm tra production ML awareness.
- **Trả lời chuẩn:** Cần: (1) preprocessor.pkl — scaler fitted trên train, (2) threshold.pkl — optimal decision boundary, (3) Feature engineering code — logic tính 15 derived features (add_derived_features function), (4) Schema validation — input phải có đúng 14 raw features, đúng dtype, (5) Monitoring — data drift detection, model performance monitoring, (6) Fallback khi model fail, (7) API endpoint (FastAPI/Flask) + Docker, (8) A/B testing so với old system.
- **Thường sai ở:** Chỉ nói "upload model.pkl lên server", quên preprocessor và feature engineering code.
- **Trả lời nhanh:** Cần: model + preprocessor + threshold + feature engineering code + schema validation + monitoring. Thiếu bất kỳ thành phần = inference sai hoặc crash.
- **Mức độ:** Hard

---

## 18. BUSINESS RECOMMENDATION

### Câu 18.1
- **Câu hỏi:** Giải thích KB1 và KB2 — 2 kịch bản business. Tại sao dùng 2 threshold khác nhau?
- **Mục đích:** Kiểm tra khả năng translate model output thành business decision.
- **Trả lời chuẩn:** KB1 (Stable Operations): LightGBM θ*=0.5041 — tối ưu F1, cân bằng Precision/Recall, total cost 100.9M VND. Phù hợp khi muốn ROI tốt nhất với ngân sách ổn định. KB2 (Growth Campaign): XGBoost θ*=0.3892 — threshold thấp hơn → Recall cao hơn → bắt được nhiều khách tái mua → FP nhiều hơn nhưng total cost 94.5M VND (thấp hơn KB1) vì giảm được FN (150K/person) nhiều hơn FP (30K/person) phát sinh. Phù hợp chiến dịch mở rộng thị phần.
- **Thường sai ở:** Không giải thích được tại sao KB2 total cost thấp hơn KB1 dù aggressive hơn.
- **Trả lời nhanh:** KB1 = balanced F1, stable ops. KB2 = recall-focused, aggressive → FN ít hơn (150K) bù FP nhiều hơn (30K) → net cost thấp hơn vì FN đắt gấp 5×.
- **Mức độ:** Hard

---

### Câu 18.2
- **Câu hỏi:** Concept drift là gì? Nếu xảy ra sau khi deploy, model bị ảnh hưởng thế nào?
- **Mục đích:** Kiểm tra hiểu biết về model degradation in production.
- **Trả lời chuẩn:** Concept drift = relationship giữa features và target thay đổi theo thời gian. Ví dụ: sau thay đổi thói quen mua sắm online → pattern cũ không valid. Triệu chứng: F1/AUC giảm dần trên production data. Xử lý: (1) Monitor metrics liên tục, (2) Retrain định kỳ với data mới, (3) Alert khi performance drop > threshold. LightGBM/XGBoost cần retrain hoàn toàn (không native online learning). Thiết kế retrain pipeline từ đầu.
- **Thường sai ở:** Không biết concept drift, không đề cập monitoring strategy.
- **Trả lời nhanh:** Concept drift → performance giảm dần. Cần monitor metrics + retrain định kỳ với data mới. Không có fix nào khác ngoài retrain.
- **Mức độ:** Hard

---

## 19. ADVANCED ML THEORY

### Câu 19.1
- **Câu hỏi:** Gradient Boosting hoạt động như thế nào? Tại sao gọi là "boosting"?
- **Mục đích:** Kiểm tra kiến thức về ensemble methods.
- **Trả lời chuẩn:** Boosting: xây dựng sequence of weak learners (shallow decision trees), mỗi tree học từ residuals của tree trước. Cụ thể: (1) Train tree₁ trên data, (2) Tính residuals = actual − predicted, (3) Train tree₂ trên residuals, (4) New prediction = prediction₁ + learning_rate × prediction₂, lặp n_estimators lần. "Boosting" = mỗi iteration boost performance bằng cách focus vào samples bị predict sai. Khác với Bagging (Random Forest) = parallel trees trên bootstrap samples.
- **Thường sai ở:** Nhầm Boosting với Bagging/Random Forest.
- **Trả lời nhanh:** Mỗi tree học residuals của tree trước → tích lũy corrections → reduce bias. learning_rate kiểm soát contribution mỗi tree. Khác Bagging là sequential not parallel.
- **Mức độ:** Hard

---

### Câu 19.2
- **Câu hỏi:** Cross-validation là gì? Tại sao project dùng hold-out thay vì k-fold?
- **Mục đích:** Kiểm tra kiến thức về validation strategies.
- **Trả lời chuẩn:** K-fold CV: chia data thành k folds, train k lần, mỗi lần 1 fold làm validation → performance estimate robust hơn. Hold-out: 1 lần split → nhanh hơn nhưng estimate noisier. Project dùng hold-out: (1) 61K samples đủ lớn → estimate ổn định, (2) Early stopping cần fixed val set để monitor — k-fold incompatible với early stopping, (3) Threshold optimization cần val set riêng biệt. Có thể thêm k-fold cho LR để estimate tốt hơn — improvement point.
- **Thường sai ở:** Nói "hold-out là sai, phải dùng k-fold" — không phải, tùy context.
- **Trả lời nhanh:** Hold-out vì: dataset lớn đủ, early stopping cần fixed val, threshold tune cần isolated val. k-fold incompatible với early stopping monitoring.
- **Mức độ:** Hard

---

### Câu 19.3
- **Câu hỏi:** Logistic Regression với lbfgs solver hoạt động như thế nào? Khi nào dùng lbfgs?
- **Mục đích:** Kiểm tra kiến thức về LR optimization.
- **Trả lời chuẩn:** L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) là quasi-Newton optimization — dùng approximation của inverse Hessian để tìm gradient direction. Hiệu quả hơn gradient descent thuần cho medium-size problems vì converge nhanh hơn. Dùng lbfgs khi: (1) Dataset vừa (không quá lớn), (2) Cần convergence nhanh, (3) Multiclass OK. Với 61K samples và 29 features → lbfgs phù hợp. saga hoặc sag phù hợp hơn cho very large datasets.
- **Thường sai ở:** Không biết lbfgs là gì, chỉ biết sgd/adam.
- **Trả lời nhanh:** lbfgs = quasi-Newton optimizer, converge nhanh hơn gradient descent thuần cho medium datasets. Phù hợp với 61K samples.
- **Mức độ:** Hard

---

## 20. TRICK QUESTIONS / BẪY PHẢN BIỆN

### Câu 20.1 [BẪY]
- **Câu hỏi:** "Model LightGBM đạt F1=0.76 trên test set — đã tốt chưa? Có thể deploy ngay không?"
- **Mục đích:** Bẫy sinh viên tự mãn với metric tốt.
- **Trả lời chuẩn:** F1=0.76 > KPI 0.72 → đạt KPI. Nhưng KHÔNG deploy ngay vì: (1) Chưa phân tích error — FN distribution? FP concentration? (2) Chưa test stability — performance ổn định qua time? (3) Chưa business validation, (4) Chưa monitoring infrastructure, (5) Chưa A/B test với old system, (6) Feature distribution production có khác train? Metric tốt = điều kiện CẦN, không phải ĐỦ để deploy.
- **Thường sai ở:** Nói "đạt KPI = deploy ngay" — thiếu production readiness mindset.
- **Trả lời nhanh:** Đạt KPI = điều kiện cần, không đủ. Cần thêm error analysis, stability test, business validation, monitoring, A/B testing.
- **Mức độ:** Very Hard

---

### Câu 20.2 [BẪY]
- **Câu hỏi:** "total_interactions vẫn xuất hiện trong code NB2. Vậy model có dùng total_interactions không?"
- **Mục đích:** Bẫy sinh viên không phân biệt EDA pipeline và Model pipeline.
- **Trả lời chuẩn:** KHÔNG. total_interactions xuất hiện trong NB2 (EDA) TRƯỚC feature selection — chỉ để analyze distribution và detect outliers, không train model. Trong NB3, total_interactions bị drop khỏi ALL_FEATURE_COLS ngay từ đầu vì chứa purchase_count (data leakage). Model chỉ train trên 14 raw features + 15 derived features, không có total_interactions hay purchase_count.
- **Thường sai ở:** Panic và không phân biệt được EDA analysis vs model training data.
- **Trả lời nhanh:** EDA analysis ≠ model features. NB2 analyze để understand, NB3 drop ngay trong feature definition. Model guaranteed không dùng total_interactions.
- **Mức độ:** Very Hard

---

### Câu 20.3 [BẪY]
- **Câu hỏi:** "Bạn impute avg_rating bằng global median TRƯỚC khi split. Đây có phải data leakage không?"
- **Mục đích:** Bẫy về preprocessing leakage tinh vi.
- **Trả lời chuẩn:** Về mặt lý thuyết là micro-leakage nhẹ — median tính từ toàn dataset bao gồm val/test → imputer "biết" một phần distribution của val/test. Trong project này tác động rất nhỏ vì median là robust statistic và distribution avg_rating ổn định. Best practice: fit imputer trên X_train only → transform val/test. Đây là limitation đã được identify — improvement cho future work.
- **Thường sai ở:** (1) Không nhận ra potential leakage, (2) Panic và nói "tất cả kết quả sai" — overreact quá mức.
- **Trả lời nhanh:** Micro-leakage về lý thuyết, thực tế tác động rất nhỏ (median robust). Best practice là fit imputer trên train only — limitation đã identify và là improvement point rõ ràng.
- **Mức độ:** Very Hard

---

### Câu 20.4 [BẪY]
- **Câu hỏi:** "LGBM AUC=0.91 trên val nhưng 0.87 trên test — overfitting không?"
- **Mục đích:** Bẫy về generalization gap và val/test interpretation.
- **Trả lời chuẩn:** Gap 4 điểm AUC là nhỏ và EXPECTED — threshold tuning và model selection dùng val set → val metric naturally cao hơn test (optimistically biased). Overfitting nghiêm trọng khi gap > 10-15 điểm. Test 0.87 là estimate thực sự của generalization, đạt KPI (≥0.87). Không phải severe overfitting.
- **Thường sai ở:** Panic khi val > test, hoặc ngược lại không thấy vấn đề khi gap rất lớn.
- **Trả lời nhanh:** Gap 4% = expected vì val được dùng cho selection → bị optimistically biased. Test 0.87 = actual generalization = đạt KPI. Không phải severe overfitting.
- **Mức độ:** Very Hard

---

### Câu 20.5 [BẪY]
- **Câu hỏi:** "Feature importance LightGBM có feature X = 0. Drop luôn có tốt hơn không?"
- **Mục đích:** Bẫy về feature selection dựa trên built-in importance.
- **Trả lời chuẩn:** Không nhất thiết. Feature importance = 0 trong 1 run có thể do: (1) Feature thực sự không useful → nên drop, (2) Feature correlated với features khác → thông tin bị "stolen" bởi correlated feature (không nên drop cả 2), (3) Training stochasticity — chạy lại có thể importance > 0. Trước khi drop: verify với permutation importance (shuffle → measure AUC drop) hoặc recursive feature elimination. Blind drop = risky.
- **Thường sai ở:** Đồng ý "importance=0 → drop ngay" mà không xét collinearity.
- **Trả lời nhanh:** Không chắc chắn. Importance=0 có thể do collinearity, không phải feature vô dụng. Verify bằng permutation importance trước khi drop.
- **Mức độ:** Hard

---

### Câu 20.6 [BẪY]
- **Câu hỏi:** "Bạn có 14 raw features nhưng slide nói 16. Số nào đúng?"
- **Mục đích:** Kiểm tra consistency giữa slides/code và khả năng giải thích discrepancy.
- **Trả lời chuẩn:** 14 raw features là đúng trong model training. Ban đầu có 19 features (sau khi bỏ purchase_count, total_interactions ra khỏi 20 cột gốc = 18). Sau EDA, drop thêm dominant_gender và dominant_location (Chi-square p>0.05, Cramér's V≈0) → còn 16 numeric + categorical. Sau khi exclude user_id và product_category (keys) → 14 features thực sự dùng để train. Nếu slides nói 16 → có thể tính trước khi drop dominant_gender/location.
- **Thường sai ở:** Panic, không trace được logic drop từng bước.
- **Trả lời nhanh:** 14 là số features thực sự train. 16 có thể là trước khi drop dominant_gender/location (Chi-square p>0.05). Trace: 20 cột gốc → -2 leakage → -2 demographics EDA → -2 keys = 14.
- **Mức độ:** Hard

---

## 🏆 TOP 20 CÂU HỎI NGUY HIỂM NHẤT

> Nếu trả lời sai những câu này sẽ mất điểm nặng nhất. Ôn kỹ trước khi báo cáo.

| # | Câu hỏi | Lý do nguy hiểm |
|---|---------|-----------------|
| 1 | Tại sao purchase_count phải xóa? | Core leakage — sai là fail |
| 2 | total_interactions chứa gì? Tại sao xóa? | Indirect leakage — nhiều người bỏ qua |
| 3 | Otsu T*=4.02 tính như thế nào? | Core methodology |
| 4 | Tại sao không dùng Q2/median thay Otsu? | Must justify choice |
| 5 | Scaler fit trên gì, transform trên gì? | Classic leakage trap |
| 6 | Threshold tối ưu tìm trên val hay test? | Data snooping trap |
| 7 | cart_rate > 1 là lỗi hay hợp lệ? | Domain knowledge test |
| 8 | dominant_gender/location bị drop tại sao? | Must have statistical evidence |
| 9 | PR-AUC vs ROC-AUC cho imbalanced data | Advanced metric knowledge |
| 10 | LR cần scale, tree không — tại sao? | Algorithm mechanics |
| 11 | pure_beh không có purchase_count — tại sao? | Leakage awareness in FE |
| 12 | +1 vào mẫu số có tên kỹ thuật gì? | Laplace smoothing |
| 13 | scale_pos_weight = bao nhiêu, tính sao? | Imbalance handling |
| 14 | LightGBM leaf-wise vs XGBoost level-wise | Tree growing strategy |
| 15 | early_stopping_rounds monitor trên tập nào? | Pipeline correctness |
| 16 | KB1 vs KB2 — threshold khác nhau tại sao? | Business case |
| 17 | avg_rating impute pre-split — có leakage không? | Micro-leakage trap |
| 18 | SHAP vs built-in importance — khác gì? | Interpretability depth |
| 19 | Tại sao split 70/15/15 không phải 80/20? | Val set purpose |
| 20 | SMOTE có nên dùng không? Áp dụng trên tập nào? | SMOTE + leakage trap |

---

## ❌ NHỮNG LỖI TRẢ LỜI THƯỜNG LÀM MẤT ĐIỂM

### Lỗi kỹ thuật
- **Lỗi 1:** Nói "cart_rate > 1 là lỗi dữ liệu, cần clip" → sai domain understanding
- **Lỗi 2:** Tune threshold trên test set thay vì val set → data snooping
- **Lỗi 3:** Fit scaler trên toàn dataset trước split → preprocessing leakage
- **Lỗi 4:** Nói "SMOTE áp dụng trên toàn dataset" → sai, chỉ train
- **Lỗi 5:** Nhầm total_interactions vẫn còn trong model features
- **Lỗi 6:** Nói "accuracy cao → model tốt" với imbalanced data
- **Lỗi 7:** Không biết scale_pos_weight = neg/pos ratio
- **Lỗi 8:** Nói LightGBM level-wise, XGBoost leaf-wise (ngược!)
- **Lỗi 9:** Nói "SHAP = feature importance" — SHAP là additive attribution khác hoàn toàn

### Lỗi business
- **Lỗi 10:** Không nhớ FN cost (150K) vs FP cost (30K) → không justify threshold
- **Lỗi 11:** Nói "KPI đạt → deploy ngay" → thiếu production mindset
- **Lỗi 12:** Không giải thích unit of analysis là cặp (user, category)
- **Lỗi 13:** Không kết nối Otsu T*=4.02 với business meaning (mua ≥5 lần = tái mua)

### Lỗi thống kê
- **Lỗi 14:** Giải thích sai công thức Otsu (ω₀·ω₁·(μ₀−μ₁)²)
- **Lỗi 15:** Không biết Point-Biserial là special case của Pearson
- **Lỗi 16:** Nói ROC-AUC luôn tốt hơn PR-AUC — sai với imbalanced data
- **Lỗi 17:** Panic khi val metric > test metric (là expected behavior)

---

## 💡 CÁCH TRẢ LỜI NHƯ SENIOR ML ENGINEER

### Framework STAR-ML:
1. **State the principle** — Nêu nguyên tắc chung (1 câu)
2. **Apply to project** — Áp dụng vào project cụ thể (2-3 câu)
3. **Acknowledge limitations** — Nói được limitation/trade-off (1 câu)

**Ví dụ — "Tại sao dùng Otsu?"**
- ❌ Junior: *"Vì Otsu tốt hơn Q2"*
- ✅ Senior: *"Otsu tối đa hóa inter-class variance — tìm T* phân tách 2 nhóm rõ nhất theo phân phối thực tế. Trong project, T*=4.02 cho class distribution 67/33 tự nhiên từ dữ liệu, không bias bởi domain assumption. Limitation: Otsu assume phân phối bimodal — nếu unimodal thì T* ít meaningful hơn."*

### Phrases của Senior:
- *"Trade-off giữa X và Y trong context này là..."*
- *"Best practice là... tuy nhiên trong scope project chúng tôi dùng... vì..."*
- *"Đây là limitation đã được identify — cải thiện bằng cách..."*
- *"Quyết định này dựa trên data-driven evidence từ [Chi-square/VIF/Point-Biserial]..."*
- *"Trong production scenario, điều này có nghĩa là..."*

---

## ✅ CHECKLIST TRƯỚC KHI LÊN BÁO CÁO

### Phải thuộc lòng:
- [ ] Otsu T* = 4.02, công thức ω₀·ω₁·(μ₀−μ₁)²
- [ ] purchase_count xóa vì leakage (100% accuracy giả nếu giữ)
- [ ] total_interactions = view+click+cart+wishlist+**purchase_count** → xóa indirect leakage
- [ ] 14 raw features (không có purchase_count, total_interactions, dominant_gender, dominant_location)
- [ ] 15 derived features với Laplace smoothing (+1 ở mẫu số)
- [ ] pure_beh = view+click+cart+wishlist (KHÔNG có purchase_count)
- [ ] Split 70/15/15, stratified, scaler **fit trên train only**
- [ ] LR cần scale; XGB/LGBM không cần
- [ ] Threshold tune trên **VAL set**, report trên **TEST set**
- [ ] scale_pos_weight = 41,507/20,221 ≈ 2.05 cho XGBoost
- [ ] KB1: LGBM θ=0.5041, cost 100.9M VND (stable ops)
- [ ] KB2: XGBoost θ=0.3892, cost 94.5M VND (growth campaign)
- [ ] FN cost = 150,000 VND, FP cost = 30,000 VND (FN **5× đắt hơn**)
- [ ] KPI: F1 ≥ 0.72 | ROC-AUC ≥ 0.87 | Recall ≥ 0.68
- [ ] cart_rate > 1 = HỢP LỆ (direct-to-cart)
- [ ] dominant_gender/location drop: Chi-square p > 0.05, Cramér's V ≈ 0
- [ ] avg_rating 28% NaN → impute median 4.0 (limitation: nên fit trên train only)
- [ ] LightGBM **leaf-wise**, XGBoost **level-wise**
- [ ] PR-AUC honest hơn ROC-AUC cho imbalanced data
- [ ] SMOTE chỉ apply trên TRAIN SET nếu dùng

---

## 🔄 CÂU HỎI TIẾP THEO SAU KHI BẠN TRẢ LỜI

| Nếu bạn nói... | Giảng viên thường hỏi tiếp... |
|----------------|-------------------------------|
| "Dùng F1 vì imbalanced" | "Imbalance ratio bao nhiêu? Có cần SMOTE không?" |
| "Otsu tìm T*=4.02" | "Giải thích công thức Otsu? Tại sao không dùng median?" |
| "Drop purchase_count vì leakage" | "total_interactions thì sao? Có chứa purchase_count không?" |
| "Tune threshold trên val set" | "Tại sao không test? Rồi evaluate final trên đâu?" |
| "LightGBM tốt nhất" | "Tại sao hơn XGBoost? Leaf-wise vs level-wise?" |
| "class_weight='balanced'" | "scale_pos_weight của XGBoost = bao nhiêu?" |
| "AUC = 0.87" | "AUC có ý nghĩa gì? ROC-AUC vs PR-AUC?" |
| "KB1 cost 100.9M" | "FN cost bao nhiêu? FP cost bao nhiêu? Tính thế nào?" |
| "15 derived features" | "Tại sao +1 vào mẫu số? pure_beh gồm những gì?" |
| "Impute median cho avg_rating" | "Impute trước hay sau split? Có leakage không?" |
| "Drop dominant_gender" | "Bằng chứng thống kê nào? Chi-square cho kết quả gì?" |
| "F1 đạt KPI" | "Có thể deploy ngay không? Cần làm gì thêm?" |

---

*Tài liệu chuẩn bị phản biện — Mock Project ML*
*Nhóm 2 | TranThiNgocVy/MOCK_PROJECT-NHOM2 | Tháng 5/2026*
