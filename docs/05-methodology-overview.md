# PHƯƠNG PHÁP GIẢI QUYẾT

## 3.1. Tổng quan phương pháp

Hệ thống dự đoán CTR được xây dựng dựa trên **kiến trúc học trực tuyến (Online Learning)** với ba thành phần cốt lõi:

1. **Xử lý dữ liệu luồng (Streaming Data Processing)**: Đọc và xử lý dữ liệu Terabyte mà không cần load toàn bộ vào bộ nhớ
2. **Mã hóa đặc trưng thưa (Sparse Feature Encoding)**: Sử dụng Feature Hashing để giảm chiều không gian đặc trưng từ hàng triệu xuống không gian cố định
3. **Thuật toán tối ưu trực tuyến (Online Optimization)**: FTRL-Proximal cho phép cập nhật mô hình ngay khi có dữ liệu mới và tạo ra mô hình thưa

---

## 3.2. Kiến trúc hệ thống tổng thể

Hệ thống hoạt động theo **pipeline 5 bước**:

```
File .gz (nén) 
  → [Bước 1] CriteoDataLoader: Đọc streaming từng dòng
  → [Bước 2] Preprocessor: Feature Hashing + Log Transform
  → [Bước 3] FTRL-Proximal: Dự đoán & Cập nhật online
  → [Bước 4] RunningMetrics: Theo dõi hiệu năng
  → [Bước 5] Model Save: Lưu trạng thái (z, n)
```

**Ưu điểm của kiến trúc này:**
- **Bộ nhớ cố định**: Không phụ thuộc kích thước dataset (nhờ Hashing Trick)
- **Xử lý tuần tự**: Phù hợp với thuật toán online learning (không cần Spark)
- **Cập nhật tức thì**: Mô hình học ngay từ mẫu đầu tiên, không cần đợi batch

---

## 3.3. Xử lý dữ liệu luồng (Streaming Data Processing)

### 3.3.1. Đọc dữ liệu nén không giải nén toàn bộ

**Vấn đề**: File Criteo có thể lên tới **1.6GB nén** (vài GB khi giải nén). Không thể load toàn bộ vào RAM.

**Giải pháp**: Sử dụng `gzip.open()` với mode `'rt'` để đọc **text stream**:
- Thư viện gzip giải nén **từng khối** khi đọc, không tạo file tạm trên đĩa
- Vòng lặp `for line in f` chỉ giữ **một dòng** trong bộ nhớ tại một thời điểm
- Chi phí bộ nhớ: **O(1)** không phụ thuộc kích thước file

**Kết quả**: Mỗi dòng TSV được parse thành:
- `label`: 0 hoặc 1 (click / không click)
- `features`: List 39 phần tử (13 integer + 26 categorical)

### 3.3.2. Streaming Iterator

Để phù hợp với online learning, hệ thống sử dụng `StreamingIterator`:
- Đọc **từng mẫu** một (không phải batch)
- Mỗi mẫu được xử lý ngay lập tức → mô hình cập nhật ngay
- Phù hợp với thuật toán FTRL yêu cầu cập nhật tuần tự

---

## 3.4. Tiền xử lý đặc trưng (Feature Preprocessing)

### 3.4.1. Xử lý giá trị thiếu (Missing Values)

**Chiến lược**:
- **Integer features**: Thay thế bằng `-1` (marker đặc biệt)
- **Categorical features**: Giữ nguyên chuỗi rỗng `''` hoặc token `'__MISSING__'`

**Lý do**: Trong online learning, không thể tính toán thống kê (mean, median) trước vì dữ liệu đến tuần tự. Marker `-1` cho phép mô hình học pattern "missing" như một feature riêng.

### 3.4.2. Log Transformation cho đặc trưng số

**Công thức**: \(x' = \log(1 + x)\) cho \(x \geq 0\)

**Lý do**: 
- Dữ liệu Criteo có phân phối **lệch phải** (skewed) - nhiều giá trị nhỏ, ít giá trị rất lớn
- Log transform làm phân phối **gần chuẩn hơn** → giúp mô hình học tốt hơn
- Áp dụng cho 13 integer features

### 3.4.3. Feature Hashing (Hashing Trick)

**Vấn đề**: 
- Categorical features có **cardinality cực cao** (hàng triệu giá trị unique)
- Không thể tạo one-hot encoding → không gian đặc trưng quá lớn

**Giải pháp - Feature Hashing**:
- **Băm** mỗi feature `(name, value)` thành một **bucket index** trong không gian cố định
- Công thức: \(h(\text{feature}) = \text{hash}(\text{name} + ":" + \text{value}) \bmod N\)
- \(N\): số bucket (ví dụ \(2^{18} = 262,144\))

**Ưu điểm**:
- **Bộ nhớ cố định**: Chỉ cần \(N\) trọng số, không phụ thuộc số lượng giá trị unique
- **O(1) lookup**: Không cần dictionary tra cứu
- **Xử lý unseen features**: Feature mới tự động được hash vào bucket

**Signed Hashing** (tùy chọn):
- Thêm dấu \(+1\) hoặc \(-1\) dựa trên hash phụ
- Giảm **bias do collision** khi nhiều feature hash vào cùng bucket

**Kết quả**: Vector đặc trưng thưa `Dict[int, float]` với:
- Key: bucket index (0 đến N-1)
- Value: giá trị feature (có thể tích lũy nếu collision)

---

## 3.5. Thuật toán FTRL-Proximal

### 3.5.1. Bài toán tối ưu

**Mục tiêu**: Tối thiểu hóa **Log-Loss** trên dữ liệu streaming:

\[
\min_w \sum_{t=1}^T \ell_t(w_t) + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2
\]

Trong đó:
- \(\ell_t(w_t) = -y_t \log p_t - (1-y_t)\log(1-p_t)\): Log-loss tại thời điểm \(t\)
- \(p_t = \sigma(w_t \cdot x_t) = \frac{1}{1+e^{-w_t \cdot x_t}}\): Xác suất dự đoán
- \(\lambda_1\): L1 regularization (tạo sparsity)
- \(\lambda_2\): L2 regularization (tránh overfitting)

### 3.5.2. Trạng thái của thuật toán

FTRL-Proximal duy trì **hai biến trạng thái** cho mỗi feature \(i\):

- **\(z_i\)**: Tổng gradient đã điều chỉnh (adjusted gradient sum)
- **\(n_i\)**: Tổng bình phương gradient (sum of squared gradients)

**Khác biệt với SGD**: Thay vì lưu trực tiếp trọng số \(w_i\), FTRL lưu \((z_i, n_i)\) và tính \(w_i\) "lười" (lazy) khi cần.

### 3.5.3. Vòng lặp học trực tuyến (Online Learning Loop)

Với mỗi mẫu \((x_t, y_t)\):

#### Bước 1: Dự đoán (Prediction)
- Tính trọng số hiện tại \(w_{t,i}\) từ \((z_i, n_i)\) (xem công thức ở Bước 4)
- Tính xác suất: \(p_t = \sigma(\sum_i w_{t,i} x_{t,i})\)

#### Bước 2: Tính gradient
- Gradient tổng: \(g_t = p_t - y_t\)
- Gradient theo từng feature: \(g_{t,i} = g_t \cdot x_{t,i}\)

#### Bước 3: Cập nhật \(n_i\) (tích lũy thông tin)
\[
n_i \leftarrow n_i + g_{t,i}^2
\]

**Ý nghĩa**: \(n_i\) đo lường "đã học bao nhiêu" về feature \(i\). Feature xuất hiện nhiều lần với gradient lớn → \(n_i\) lớn → learning rate nhỏ hơn (ổn định hơn).

#### Bước 4: Tính \(\sigma_i\) (learning rate factor)
\[
\sigma_i = \frac{\sqrt{n_i^{\text{mới}}} - \sqrt{n_i^{\text{cũ}}}}{\alpha}
\]

**Ý nghĩa**: Sự thay đổi trong \(\sqrt{n_i}\) chia cho \(\alpha\) → điều chỉnh mức độ cập nhật dựa trên lịch sử học.

#### Bước 5: Cập nhật \(z_i\) (tích lũy gradient điều chỉnh)
\[
z_i \leftarrow z_i + g_{t,i} - \sigma_i w_{t,i}
\]

**Trực giác**:
- `+ g_{t,i}`: Thêm thông tin gradient mới
- `- σ_i w_{t,i}`: Hiệu chỉnh theo trọng số hiện tại và lịch sử → gắn với regularization

### 3.5.4. Công thức tính trọng số (Closed-form Solution)

Khi cần dùng \(w_i\) (ở bước dự đoán), FTRL tính theo công thức:

**Nếu \(|z_i| \leq \lambda_1\)**:
\[
w_i = 0
\]

**Ngược lại**:
\[
w_i = -\frac{z_i - \text{sgn}(z_i) \lambda_1}{(\beta + \sqrt{n_i})/\alpha + \lambda_2}
\]

**Giải thích**:
- **L1 regularization**: Nếu \(|z_i|\) không đủ lớn vượt ngưỡng \(\lambda_1\) → đặt \(w_i = 0\) → **sparsity**
- **Per-coordinate learning rate**: \((\beta + \sqrt{n_i})/\alpha\) → mỗi feature có learning rate riêng, giảm dần khi đã học nhiều
- **L2 regularization**: \(+ \lambda_2\) trong mẫu số → kéo trọng số về gần 0

**Kết quả**: Mô hình **rất thưa** (nhiều \(w_i = 0\)) → tiết kiệm bộ nhớ và dự đoán nhanh.

### 3.5.5. Tham số của FTRL-Proximal

- **\(\alpha\) (alpha)**: Learning rate cơ bản. Cao → học nhanh nhưng có thể không ổn định
- **\(\beta\) (beta)**: Điều chỉnh learning rate giai đoạn đầu. Thường \(\beta = 1\)
- **\(\lambda_1\) (L1)**: Ngưỡng sparsity. Cao → mô hình thưa hơn nhưng có thể mất accuracy
- **\(\lambda_2\) (L2)**: Regularization strength. Cao → trọng số nhỏ hơn, tránh overfitting

**Giá trị mặc định trong dự án**: \(\alpha = 0.1\), \(\beta = 1.0\), \(\lambda_1 = 1.0\), \(\lambda_2 = 1.0\)

---

## 3.6. So sánh với Online Logistic Regression (Baseline)

### 3.6.1. Online Logistic Regression

**Cập nhật trực tiếp**:
\[
w_{t+1,i} = w_{t,i} - \eta_t (g_{t,i} + \lambda_2 w_{t,i})
\]

Trong đó:
- \(\eta_t = \eta_0 / \sqrt{t}\): Learning rate giảm dần theo thời gian (giống nhau cho mọi feature)
- Chỉ có **L2 regularization**, không có L1

**Nhược điểm**:
- **Không tạo sparsity**: Trọng số hiếm khi bằng 0 → mô hình **đặc**, tốn RAM
- **Learning rate chung**: Không tận dụng được feature xuất hiện với tần suất khác nhau

### 3.6.2. Ưu điểm của FTRL-Proximal

1. **Sparsity**: L1 regularization → nhiều trọng số = 0 → mô hình nhẹ, dự đoán nhanh
2. **Per-coordinate learning rate**: Mỗi feature có learning rate riêng → học hiệu quả hơn
3. **Độ chính xác tốt hơn**: Trong thực nghiệm, FTRL đạt accuracy tốt hơn hoặc tương đương với OGD nhưng với mô hình thưa hơn nhiều

**Kết quả thực nghiệm** (theo paper Google):
- FTRL-Proximal: **baseline** (độ chính xác và sparsity)
- RDA: +3% non-zero weights, -0.6% accuracy
- FOBOS: +38% non-zero weights, accuracy tương đương
- OGD-Count: +216% non-zero weights, accuracy tương đương

---

## 3.7. Quy trình huấn luyện (Training Pipeline)

### 3.7.1. StreamingTrainer

**Chức năng**: Điều phối toàn bộ quá trình huấn luyện

**Vòng lặp chính**:
```
For mỗi mẫu (label, raw_features) từ StreamingIterator:
  1. Preprocess: raw_features → sparse_features (Dict[int, float])
  2. Predict & Update: model.update(sparse_features, label) → nhận prediction
  3. Metrics: RunningMetrics.update(label, prediction)
  4. Logging: Nếu đạt log_interval → in trạng thái (Loss, Accuracy, Sparsity)
```

**Đặc điểm**:
- **Xử lý tuần tự**: Một mẫu tại một thời điểm (true online learning)
- **Bộ nhớ cố định**: Không load dataset vào RAM
- **Theo dõi liên tục**: Metrics được cập nhật running average

### 3.7.2. Đánh giá (Evaluation)

**Progressive Validation**: Đánh giá trên tập test cũng theo streaming:
- Đọc từng mẫu test
- Dự đoán với mô hình hiện tại (không cập nhật)
- Tính Log-Loss, Accuracy, AUC

**Lợi ích**: Có thể đánh giá trên tập test lớn mà không cần load vào bộ nhớ.

---

## 3.8. Lưu trữ mô hình (Model Persistence)

### 3.8.1. Cấu trúc file `.pkl`

Mô hình được lưu dưới dạng dictionary:
- **`z`**: Dict[int, float] - Trạng thái gradient sum cho mỗi feature
- **`n`**: Dict[int, float] - Trạng thái squared gradient sum
- **Hyperparameters**: \(\alpha, \beta, \lambda_1, \lambda_2\)
- **Metadata**: `num_updates` (số mẫu đã học)

### 3.8.2. Incremental Learning

Nhờ lưu trạng thái \((z, n)\), mô hình có thể:
- **Tiếp tục học**: Load model → train thêm trên dữ liệu mới
- **Phục vụ dự đoán**: Load model → predict mà không cần train lại

**Lợi ích**: Phù hợp với môi trường production nơi dữ liệu mới liên tục đến.

---

## 3.9. Tóm tắt phương pháp

| Thành phần | Kỹ thuật | Mục đích |
|------------|----------|----------|
| **Data Loading** | Streaming với gzip | Đọc Terabyte không giải nén toàn bộ |
| **Preprocessing** | Feature Hashing + Log Transform | Giảm chiều không gian, xử lý missing values |
| **Algorithm** | FTRL-Proximal | Online learning với sparsity và per-coordinate learning rate |
| **Training** | Streaming Trainer | Cập nhật từng mẫu, theo dõi metrics liên tục |
| **Evaluation** | Progressive Validation | Đánh giá trên streaming data |

**Độ phức tạp**:
- **Thời gian**: O(số feature không-zero) cho mỗi mẫu
- **Bộ nhớ**: O(số bucket hash) = cố định, không phụ thuộc dataset size

**Ưu điểm tổng thể**:
- Xử lý được dataset Terabyte trên máy tính cá nhân
- Mô hình thưa → tiết kiệm bộ nhớ, dự đoán nhanh
- Cập nhật online → phù hợp với dữ liệu streaming thời gian thực
