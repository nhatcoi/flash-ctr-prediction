# BÁO CÁO BÀI TẬP LỚN
## Môn: THUẬT TOÁN ỨNG DỤNG

---

| Thông tin         | Nội dung                                                              |
|-------------------|-----------------------------------------------------------------------|
| **Đề tài**        | Bài toán chọn quảng cáo hiển thị trên log cực lớn (1TB Click Logs)    |
| **Sinh viên**     | [ Họ và tên sinh viên ]                                               |
| **MSSV**          | [ Mã số sinh viên ]                                                   |
| **Lớp**           | [ Tên lớp ]                                                           |
| **Giáo viên HD**  | [ Họ và tên giáo viên hướng dẫn ]                                     |
| **Ngày nộp**      | [ Ngày/Tháng/Năm ]                                                    |

---

## MỤC LỤC

1. [Giới thiệu bài toán](#1-giới-thiệu-bài-toán)
2. [Mô tả dữ liệu](#2-mô-tả-dữ-liệu)
3. [Phương pháp giải quyết](#3-phương-pháp-giải-quyết)
4. [Kiến trúc hệ thống](#4-kiến-trúc-hệ-thống)
5. [Kết quả thực nghiệm](#5-kết-quả-thực-nghiệm)
6. [Kết luận & Hướng phát triển](#6-kết-luận--hướng-phát-triển)
7. [Tài liệu tham khảo](#7-tài-liệu-tham-khảo)

---

## 1. GIỚI THIỆU BÀI TOÁN

### 1.1. Đặt vấn đề
Trong lĩnh vực quảng cáo trực tuyến, việc dự đoán xác suất người dùng nhấp chuột vào một quảng cáo (Click-Through Rate - CTR) là bài toán then chốt. Một hệ thống đề xuất quảng cáo hiệu quả cần phải:
- Chọn được tập quảng cáo phù hợp nhất để hiển thị cho mỗi người dùng.
- Xử lý được khối lượng dữ liệu khổng lồ (hàng Terabyte) trong thời gian thực.
- Thích nghi với sự thay đổi hành vi người dùng theo thời gian (Concept Drift).

### 1.2. Mục tiêu
**Tối đa hóa hiệu quả hiển thị quảng cáo** bằng cách xây dựng mô hình máy học có khả năng:
1. Dự đoán chính xác xác suất click cho mỗi cặp (người dùng, quảng cáo).
2. Hoạt động theo cơ chế **xử lý luồng (Streaming)** - không cần nạp toàn bộ dữ liệu vào bộ nhớ.
3. Tạo ra mô hình **thưa (Sparse)** để tối ưu tài nguyên lưu trữ và tốc độ dự đoán.

### 1.3. Thách thức kỹ thuật
| Thách thức              | Mô tả                                                                                                   |
|-------------------------|---------------------------------------------------------------------------------------------------------|
| **Quy mô dữ liệu**      | Bộ dữ liệu Criteo có dung lượng ~1.3TB (hơn 4 tỷ bản ghi), không thể nạp vào RAM máy tính thông thường.  |
| **Độ thưa dữ liệu**     | Các biến phân loại (Categorical) có hàng triệu giá trị khác nhau (High Cardinality).                     |
| **Cập nhật liên tục**   | Mô hình cần học từ dữ liệu mới liên tục (Online Learning) mà không cần huấn luyện lại từ đầu.            |

---

## 2. MÔ TẢ DỮ LIỆU

### 2.1. Nguồn dữ liệu
- **Tên bộ dữ liệu:** Criteo 1TB Click Logs Dataset
- **Nguồn:** Criteo AI Lab (Hugging Face)
- **Link:** [https://huggingface.co/datasets/criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs)

### 2.2. Mô tả chi tiết
Bộ dữ liệu bao gồm **24 tệp tin** tương ứng với **24 ngày** ghi nhận log quảng cáo (từ `day_2.gz` đến `day_23.gz`). Mỗi tệp có dung lượng khoảng **1.6GB** (đã nén).

**Cấu trúc mỗi bản ghi (40 cột):**

| STT   | Tên trường         | Kiểu dữ liệu | Mô tả                                                                 |
|-------|--------------------|--------------|-----------------------------------------------------------------------|
| 1     | `label`            | Integer      | Nhãn: 1 = Click, 0 = Không click                                      |
| 2-14  | `I1` - `I13`       | Integer      | 13 biến số (numerical) - thường là các giá trị đếm, tần suất          |
| 15-40 | `C1` - `C26`       | String (Hash)| 26 biến định danh (categorical) - đã được băm để bảo mật              |

> **Lưu ý:** Các giá trị thiếu (missing) được biểu diễn bằng ô trống trong file TSV.

### 2.3. Thống kê dữ liệu sử dụng trong thực nghiệm
| Tệp dữ liệu  | Vai trò          | Số mẫu sử dụng | Ghi chú                          |
|--------------|------------------|----------------|----------------------------------|
| `day_2.gz`   | Huấn luyện       | 1,000,000      | Dữ liệu ngày 2                   |
| `day_3.gz`   | Kiểm thử         | 200,000        | Dữ liệu ngày kế tiếp (Validation)|
| `day_23.gz`  | Test Concept Drift | [ ... ]      | Dữ liệu cuối cùng để kiểm tra độ bền vững của mô hình |

[ Hình ảnh: Minh họa cấu trúc 1 dòng dữ liệu - Thêm ảnh chụp màn hình `gzcat data/day_2.gz | head -n 3` ]

---

## 3. PHƯƠNG PHÁP GIẢI QUYẾT

### 3.1. Tổng quan luồng xử lý
Hệ thống được thiết kế theo mô hình **Streaming Pipeline** với 4 giai đoạn chính:

[ Hình ảnh: Sơ đồ quy trình - Sử dụng file `project_workflow.puml` đã tạo ]

### 3.2. Tiền xử lý dữ liệu (Preprocessing)

#### 3.2.1. Xử lý giá trị thiếu (Missing Value Handling)
- **Biến số (Numerical):** Thay thế bằng giá trị marker `-1`.
- **Biến định danh (Categorical):** Thay thế bằng token đặc biệt `__MISSING__`.

#### 3.2.2. Biến đổi Log (Log Transformation)
Áp dụng cho các biến số để giảm độ lệch phân phối:
$$x' = \begin{cases} \log(1 + x) & \text{nếu } x \geq 0 \\ -\log(1 - x) & \text{nếu } x < 0 \end{cases}$$

#### 3.2.3. Hashing Trick (Feature Hashing)
Đây là kỹ thuật **then chốt** cho phép xử lý dữ liệu 1TB trên máy tính cá nhân:
- **Vấn đề:** Với hàng triệu giá trị categorical, nếu dùng One-Hot Encoding thì vector đặc trưng sẽ có hàng triệu chiều.
- **Giải pháp:** Sử dụng hàm băm (hash function) để ánh xạ mọi feature vào một không gian vector **kích thước cố định** ($2^{18} = 262,144$ buckets).
- **Ưu điểm:** 
  - Bộ nhớ sử dụng cố định $O(1)$ bất kể số lượng feature.
  - Không cần lưu từ điển mapping giữa tên feature và index.

**Công thức:**
$$\text{bucket\_index} = \text{hash}(\text{feature\_name} + ":" + \text{feature\_value}) \mod N$$

Trong đó $N = 2^{18}$ là số buckets.

### 3.3. Thuật toán FTRL-Proximal

#### 3.3.1. Giới thiệu
**FTRL-Proximal (Follow-The-Regularized-Leader)** là thuật toán Online Learning được Google công bố năm 2013, được thiết kế đặc biệt cho bài toán CTR trên dữ liệu lớn.

**Tài liệu tham khảo:** McMahan et al., "Ad Click Prediction: a View from the Trenches" (2013)
- Link: [https://research.google/pubs/pub41159/](https://research.google/pubs/pub41159/)

#### 3.3.2. Ý tưởng chính
- Kết hợp **Online Gradient Descent** với **Proximal Regularization**.
- Sử dụng cả **L1** và **L2 regularization** để tạo ra mô hình thưa (sparse model).
- Cập nhật trọng số theo từng tọa độ (per-coordinate update) thay vì toàn bộ vector.

#### 3.3.3. Công thức cập nhật
Với mỗi tọa độ $i$ (tương ứng với một feature):

**Bước 1:** Tính gradient
$$g_i^{(t)} = (p^{(t)} - y^{(t)}) \cdot x_i^{(t)}$$

**Bước 2:** Cập nhật biến tích lũy
$$n_i^{(t)} = n_i^{(t-1)} + (g_i^{(t)})^2$$
$$z_i^{(t)} = z_i^{(t-1)} + g_i^{(t)} - \sigma^{(t)} \cdot w_i^{(t)}$$

**Bước 3:** Tính trọng số mới (Closed-form solution)
$$w_i^{(t+1)} = \begin{cases} 0 & \text{nếu } |z_i^{(t)}| \leq \lambda_1 \\ -\frac{z_i^{(t)} - \text{sign}(z_i^{(t)}) \cdot \lambda_1}{\lambda_2 + \frac{\beta + \sqrt{n_i^{(t)}}}{\alpha}} & \text{ngược lại} \end{cases}$$

Trong đó:
- $\alpha, \beta$: Tham số learning rate
- $\lambda_1, \lambda_2$: Hệ số regularization L1 và L2
- $\sigma^{(t)} = \frac{\sqrt{n_i^{(t)}} - \sqrt{n_i^{(t-1)}}}{\alpha}$

#### 3.3.4. Tính chất quan trọng: Sparsity
Điều kiện để $w_i = 0$ là $|z_i| \leq \lambda_1$. Điều này có nghĩa:
- Nếu một feature không đóng góp đủ mạnh cho việc dự đoán, trọng số của nó sẽ bị ép về **đúng bằng 0**.
- Kết quả: Mô hình cuối cùng chỉ chứa một tập con nhỏ các feature quan trọng, giúp **tiết kiệm bộ nhớ** và **tăng tốc độ dự đoán**.

### 3.4. Thuật toán so sánh: Online Logistic Regression
Để đánh giá hiệu quả của FTRL, dự án sử dụng **Online Logistic Regression** với Stochastic Gradient Descent (SGD) làm baseline.

| Tiêu chí             | FTRL-Proximal                     | Online Logistic Regression        |
|----------------------|-----------------------------------|-----------------------------------|
| Regularization       | L1 + L2 (Proximal)                | L2 only                           |
| Sparsity             | Cao (~85-90%)                     | Không có (0%)                     |
| Độ phức tạp không gian | $O(\text{non-zero weights})$   | $O(\text{all weights})$           |
| Tốc độ hội tụ        | Nhanh hơn trên dữ liệu thưa       | Chậm hơn                          |

---

## 4. KIẾN TRÚC HỆ THỐNG

Hệ thống được thiết kế theo **kiến trúc xử lý luồng (streaming pipeline)** với ba đặc tính then chốt cho bài toán dữ liệu 1TB:

| Đặc tính | Mô tả |
|----------|--------|
| **Fixed-Memory** | Bộ nhớ sử dụng bị chặn bởi hằng số (số bucket, kích thước batch), không tăng theo dung lượng file hay số mẫu. |
| **One-Pass** | Dữ liệu chỉ đi qua pipeline đúng một lần; không cần đọc lại toàn bộ dataset. |
| **Online Learning** | Mô hình cập nhật từng mẫu một; kết quả cuối là file mô hình nhỏ (.pkl), không phải bản sao dữ liệu. |

---

### 4.1. Tổng quan kiến trúc theo tầng

Luồng xử lý đi qua bốn tầng chính:

1. **Tầng dữ liệu:** Đọc file `.gz`/`.txt` theo luồng (CriteoDataLoader, StreamingIterator).
2. **Tầng tiền xử lý:** Xử lý thiếu, log transform, Feature Hashing → vector thưa `{bucket: value}`.
3. **Tầng thuật toán:** FTRL-Proximal (chính) và Online Logistic Regression (so sánh).
4. **Tầng huấn luyện & đánh giá:** StreamingTrainer điều phối; RunningMetrics, Visualizer xuất biểu đồ; mô hình lưu `.pkl`.

---

### 4.2. Cấu trúc thư mục dự án

```
TTUD/
├── data/
│   ├── sample/          # Dữ liệu mẫu để test nhanh (.txt)
│   ├── day_2.gz         # Dữ liệu huấn luyện (Criteo)
│   └── day_3.gz         
│   └── ........         
├── src/
│   ├── data/            # Đọc và tiền xử lý dữ liệu
│   │   ├── data_loader.py    # CriteoDataLoader, StreamingIterator
│   │   └── preprocessing.py  # Preprocessor (Missing, Log, Hashing)
│   ├── algorithms/      # Thuật toán học
│   │   ├── ftrl.py           # FTRL-Proximal
│   │   └── online_logistic.py # Online LR (baseline)
│   ├── training/        # Pipeline huấn luyện
│   │   └── trainer.py        # StreamingTrainer
│   └── evaluation/      # Đo lường và trực quan hóa
│       ├── metrics.py       # RunningMetrics, log_loss, AUC
│       ├── visualizer.py    # Biểu đồ training, so sánh sparsity
│       └── graph_analysis.py # Phân tích đồ thị đặc trưng
├── models/              # Mô hình đã huấn luyện (.pkl)
├── outputs/             # Biểu đồ kết quả (.png)
├── docs/                # Tài liệu (giai đoạn, slide, Q&A)
├── main.py              # Entry point (--train, --evaluate, --compare, --demo)
├── project_workflow.puml # Sơ đồ PlantUML quy trình hệ thống
└── requirements.txt     # Thư viện Python
```

---

### 4.3. Các thành phần chính

#### 4.3.1. Tầng dữ liệu

| Thành phần | File | Chức năng |
|------------|------|-----------|
| **CriteoDataLoader** | `src/data/data_loader.py` | Đọc file TSV/.gz theo batch; dùng `gzip.open` + `readline()` để không nạp toàn bộ vào RAM. |
| **StreamingIterator** | `src/data/data_loader.py` | Lặp từng mẫu một (label, raw_features) phục vụ học trực tuyến; hỗ trợ `max_samples`. |

**Luận điểm:** Độ phức tạp không gian tại bước đọc là $O(1)$ theo kích thước file — cho phép xử lý 1TB mà không tràn bộ nhớ.

#### 4.3.2. Tầng tiền xử lý

| Thành phần | File | Chức năng |
|------------|------|-----------|
| **Preprocessor** | `src/data/preprocessing.py` | (1) Điền giá trị thiếu (numerical: -1, categorical: `__MISSING__`); (2) Log transform cho cột số; (3) Feature Hashing → dict thưa `{bucket_index: value}` với $N = 2^{18}$ bucket. |

**Luận điểm:** Không gian đặc trưng cố định $O(N)$; không cần từ điển ID → index, giải quyết High Cardinality.

#### 4.3.3. Tầng thuật toán

| Thành phần | File | Chức năng |
|------------|------|-----------|
| **FTRLProximal** | `src/algorithms/ftrl.py` | Cập nhật trạng thái $(z_i, n_i)$ theo từng mẫu; trọng số $w_i$ tính lazy từ $z, n$; L1/L2 → mô hình thưa; `save`/`load` `.pkl`. |
| **OnlineLogisticRegression** | `src/algorithms/online_logistic.py` | SGD + L2, không sparsity; dùng để so sánh với FTRL. |

#### 4.3.4. Tầng huấn luyện & đánh giá

| Thành phần | File | Chức năng |
|------------|------|-----------|
| **StreamingTrainer** | `src/training/trainer.py` | Điều phối: StreamingIterator → Preprocessor → model.update(); tích lũy RunningMetrics; in sparsity theo `log_interval`; lưu/tải mô hình. |
| **RunningMetrics** | `src/evaluation/metrics.py` | Log-Loss, Accuracy, AUC tích lũy theo thời gian thực. |
| **Visualizer** | `src/evaluation/visualizer.py` | Vẽ đồ thị tiến trình huấn luyện, so sánh Log-Loss/Accuracy/Sparsity giữa FTRL và Online LR. |

**Entry point:** `main.py` — lệnh `--train`, `--evaluate`, `--compare`, `--demo` gọi lần lượt train, evaluate, so sánh hai mô hình, và chạy demo với dữ liệu mẫu.

---

### 4.4. Luồng dữ liệu (Data Flow)

```
File .gz/.txt  →  CriteoDataLoader / StreamingIterator  →  từng dòng (raw)
       →  Preprocessor (missing, log, hash)  →  vector thưa Dict[int, float]
       →  FTRLProximal.update(features, label)  →  cập nhật z, n; tích lũy metrics
       →  định kỳ: log sparsity, (tuỳ chọn) lưu .pkl
       →  Visualizer nhận history  →  xuất biểu đồ (outputs/)
```

Đánh giá (evaluate): Load `.pkl` → đọc từng mẫu test → `model.predict(features)` (không update) → tính Log-Loss, Accuracy, AUC.

---

### 4.5. Sơ đồ quy trình xử lý

Sơ đồ chi tiết các tầng và luồng dữ liệu được mô tả bằng PlantUML trong file `project_workflow.puml`.

[ Hình ảnh: Chèn ảnh render từ file `project_workflow.puml` — dùng PlantUML (online hoặc extension VSCode) để xuất ảnh PNG. ]

---

## 5. KẾT QUẢ THỰC NGHIỆM

Thực nghiệm được thiết kế để (1) đánh giá hiệu năng của **FTRL-Proximal** trên dữ liệu Criteo và (2) **so sánh với baseline** **Online Logistic Regression** (OLR). OLR cùng pipeline (streaming, Feature Hashing), dùng SGD + L2 nhưng **không có L1** nên mô hình **đặc** (sparsity 0%). So sánh này cho thấy lợi ích của FTRL về độ chính xác (Log-Loss) và tiết kiệm tài nguyên (sparsity).

---

### 5.1. Môi trường và thiết lập

| Thông số        | Giá trị                          |
|-----------------|----------------------------------|
| **Hệ điều hành**| macOS                            |
| **Python**      | 3.14                             |
| **RAM**         | [ ... GB ]                       |
| **CPU**         | [ ... ]                          |
| **Storage**     | NVME External SSD                |

**Tham số huấn luyện (FTRL):**

| Tham số           | Ký hiệu    | Giá trị   |
|-------------------|------------|-----------|
| Learning rate     | $\alpha$   | 0.1       |
| Smoothing         | $\beta$    | 1.0       |
| L1 Regularization| $\lambda_1$| 1.0       |
| L2 Regularization| $\lambda_2$| 1.0       |
| Hash buckets      | $N$        | $2^{18}$  |

**Dữ liệu:** Train = `day_2.gz` (Criteo), Test = `day_3.gz`; cùng tiền xử lý (Preprocessor) cho cả FTRL và Online Logistic Regression.

---

### 5.2. Kết quả huấn luyện FTRL-Proximal

**Lệnh:**
```bash
python main.py --train --data data/day_2.gz --max-samples 1000000 --output models/ftrl_big.pkl
```

**Kết quả trên tập train (1M mẫu):**

| Chỉ số                  | Giá trị                          |
|-------------------------|----------------------------------|
| Số mẫu huấn luyện       | 1,000,000                        |
| Thời gian huấn luyện    | ~84 giây                         |
| Tốc độ xử lý            | ~12,000 samples/giây             |
| Log-Loss cuối cùng      | **0.1269**                       |
| Accuracy                | **96.85%**                       |
| Sparsity (độ thưa)      | **87.75%** (30,881/252,170 non-zero) |

[ Hình ảnh: Ảnh chụp màn hình Terminal khi chạy lệnh train. ]

---

### 5.3. Kết quả đánh giá trên tập test (FTRL)

Mô hình FTRL đã huấn luyện được đánh giá trên dữ liệu ngày tiếp theo (`day_3.gz`) — không cập nhật trọng số, chỉ dự đoán.

**Lệnh:**
```bash
python main.py --evaluate --data data/day_3.gz --model models/ftrl_big.pkl --max-samples 200000
```

**Kết quả:**

| Chỉ số         | Giá trị   | Nhận xét                                      |
|----------------|-----------|-----------------------------------------------|
| **Log-Loss**   | **0.1169**| Thấp hơn train → mô hình không overfit        |
| **Accuracy**   | **97.17%**| Độ chính xác cao                              |
| **AUC**        | **0.7500**| Mức tốt cho bài toán CTR thực tế              |

---

### 5.4. So sánh FTRL-Proximal với Online Logistic Regression

Để đánh giá vai trò của **L1 regularization** và **sparsity**, dự án sử dụng **Online Logistic Regression** (OLR) làm **baseline**: cùng luồng dữ liệu, cùng Feature Hashing, nhưng OLR chỉ dùng SGD + L2, không có L1 nên **không tạo sparsity** (mọi trọng số đã cập nhật đều khác 0). Hai mô hình được huấn luyện và đánh giá trong cùng điều kiện bằng lệnh so sánh:

**Lệnh:**
```bash
python main.py --compare --data data/day_2.gz --test-data data/day_3.gz --max-samples 200000 --plot
```

**Bảng so sánh (cùng train 200k mẫu, đánh giá trên test 200k):**

| Chỉ số              | FTRL-Proximal | Online Logistic Regression | Ghi chú        |
|---------------------|---------------|----------------------------|----------------|
| **Log-Loss**        | **0.1139**    | 0.1189                     | Thấp hơn = tốt hơn |
| **Accuracy**        | 97.27%        | 97.27%                     | Tương đương   |
| **Sparsity**        | **~88%**      | 0%                         | OLR không có L1 |
| **Số trọng số ≠ 0** | ~30,000       | ~250,000                   | FTRL thưa hơn ~8× |

[ Hình ảnh: Biểu đồ so sánh Log-Loss và Accuracy — file `outputs/model_comparison.png`. ]

**Nhận xét:**

1. **Log-Loss:** FTRL-Proximal đạt **thấp hơn** OLR (0.1139 vs 0.1189), cho thấy dự đoán xác suất click chính xác hơn nhờ cơ chế proximal và per-coordinate learning rate.
2. **Sparsity:** FTRL tạo mô hình **thưa** (~88% trọng số bằng 0); Online Logistic Regression **không có sparsity** (0%), nên số trọng số khác 0 lớn hơn khoảng **8 lần**. Điều này thể hiện rõ lợi ích của việc sử dụng OLR làm baseline để so sánh.
3. **Ý nghĩa thực tiễn:** Cùng mức Accuracy, FTRL vừa cải thiện Log-Loss vừa giảm đáng kể dung lượng mô hình và tăng tốc inference, phù hợp triển khai production.

---

### 5.5. Biểu đồ trực quan

[ Hình ảnh: Biểu đồ Log-Loss theo thời gian huấn luyện — `outputs/` (nếu có). ]

[ Hình ảnh: Biểu đồ so sánh Sparsity FTRL vs Online Logistic Regression — `outputs/`. ]

---

## 6. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận

Dự án đã đạt được các mục tiêu đề ra:

| Mục tiêu | Kết quả |
|----------|---------|
| **Xử lý dữ liệu quy mô lớn** | Streaming + Feature Hashing ($2^{18}$ bucket) → bộ nhớ cố định, có thể huấn luyện trên dữ liệu 1TB mà không tràn RAM. |
| **Mô hình chính xác** | FTRL-Proximal đạt Log-Loss ~0,11–0,13 và Accuracy ~97% trên Criteo; đánh giá trên day_3 cho thấy không overfit. |
| **Tối ưu tài nguyên** | Sparsity ~88% (so sánh với Online Logistic Regression 0%) → mô hình nhẹ hơn ~8 lần, inference nhanh, phù hợp triển khai thực tế. |

**Ưu điểm chính của giải pháp:**

- **Online Learning:** Cập nhật mô hình từng mẫu; có thể học tiếp (incremental) từ file `.pkl` mà không train lại từ đầu.
- **Scalability:** Kiến trúc one-pass, fixed-memory cho phép mở rộng sang dữ liệu lớn hơn.
- **So sánh có căn cứ:** Sử dụng Online Logistic Regression làm baseline cùng pipeline → làm rõ lợi ích của FTRL (Log-Loss thấp hơn, sparsity cao).

---

### 6.2. Hướng phát triển

| Hạn chế hiện tại | Hướng phát triển đề xuất |
|------------------|---------------------------|
| **Chưa mô hình hóa tương tác đặc trưng (feature interactions)** | Bổ sung Poly2, FM (Factorization Machines) hoặc Deep & Wide để nắm bắt tương tác cặp (user–ad, v.v.). |
| **Hyperparameters (α, β, λ₁, λ₂) chọn thủ công** | Grid Search, Random Search hoặc Bayesian Optimization trên tập validation. |
| **Chạy đơn luồng trên một máy** | Song song hóa: Spark MLlib, Ray, hoặc phân tán FTRL trên nhiều worker. |
| **Chưa xử lý concept drift một cách tường minh** | Giám sát Log-Loss/AUC theo thời gian; điều chỉnh learning rate hoặc retrain định kỳ khi performance suy giảm. |

**Tóm tắt:** Dự án chứng minh có thể xây dựng hệ thống dự đoán CTR theo kiến trúc streaming + FTRL-Proximal trên dữ liệu Criteo quy mô lớn, với kết quả so sánh rõ ràng so với Online Logistic Regression. Các bước tiếp theo tập trung vào tương tác đặc trưng, tối ưu tham số và mở rộng quy mô tính toán.

---

## 7. TÀI LIỆU THAM KHẢO

1. McMahan, H. B., et al. (2013). **"Ad Click Prediction: a View from the Trenches."** Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
   - Link: [https://research.google/pubs/pub41159/](https://research.google/pubs/pub41159/)

2. Criteo AI Lab. **Criteo 1TB Click Logs Dataset.**
   - Link: [https://huggingface.co/datasets/criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs)

3. Weinberger, K., et al. (2009). **"Feature Hashing for Large Scale Multitask Learning."** Proceedings of the 26th International Conference on Machine Learning.

4. Mã nguồn dự án: [ Link GitHub nếu có ]

---

## PHỤ LỤC

### A. Hướng dẫn chạy dự án
```bash
# 1. Cài đặt môi trường
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Tải dữ liệu từ Hugging Face
hf auth login
hf download criteo/CriteoClickLogs day_2.gz day_3.gz --repo-type dataset --local-dir ./data

# 3. Huấn luyện mô hình
python main.py --train --data data/day_2.gz --max-samples 1000000 --output models/ftrl_big.pkl

# 4. Đánh giá mô hình
python main.py --evaluate --data data/day_3.gz --model models/ftrl_big.pkl --max-samples 200000

# 5. So sánh thuật toán
python main.py --compare --data data/day_2.gz --test-data data/day_3.gz --max-samples 200000 --plot
```

### B. Danh sách hình ảnh cần bổ sung
- [ ] Ảnh chụp màn hình Terminal khi chạy `train`
- [ ] Ảnh chụp màn hình Terminal khi chạy `evaluate`
- [ ] Biểu đồ so sánh từ thư mục `outputs/`
- [ ] Sơ đồ quy trình từ `project_workflow.puml`
- [ ] Ảnh minh họa cấu trúc dữ liệu

---

**--- HẾT ---**
