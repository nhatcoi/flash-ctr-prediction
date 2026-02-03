# Giai đoạn 3: Áp dụng thuật toán tối ưu FTRL-Proximal

## Vai trò trong pipeline

Sau **Giai đoạn 1** (Xử lý dữ liệu luồng) và **Giai đoạn 2** (Tiền xử lý – Mã hóa đặc trưng thưa), mỗi mẫu đã được biểu diễn dưới dạng **vector thưa** (dict bucket → value). **Giai đoạn 3** sử dụng thuật toán **FTRL-Proximal** để:

- **Cập nhật mô hình ngay** khi có dữ liệu mới (online learning).
- **Tạo ra mô hình thưa** (nhiều trọng số bằng 0), tiết kiệm bộ nhớ và tăng tốc dự đoán.

---

## Input và Output của giai đoạn 3

| | Mô tả |
|---|--------|
| **Input** | Vector thưa `Dict[int, float]` (output Feature Hashing) + nhãn `label` ∈ {0, 1} cho **một mẫu**. |
| **Output** | Xác suất dự đoán click \(p \in [0,1]\); đồng thời **cập nhật nội bộ** trạng thái mô hình \((z_i, n_i)\) cho các feature có trong mẫu. |
| **Sau nhiều mẫu** | Mô hình (trọng số \(w_i\) suy từ \(z, n\)) hội tụ; có thể lưu file `.pkl` (z, n, hyperparameters) để dự đoán hoặc học tiếp. |

---

## 1. Bài toán tối ưu

- **Mục tiêu:** Dự đoán xác suất click \(p = \sigma(w \cdot x)\) với vector thưa \(x\), tối thiểu hóa **Log-Loss** và có **regularization**:
  \[
  \min_w \sum_t \ell_t(w) + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2
  \]
- **FTRL-Proximal** giải bài toán này theo kiểu **online**: mỗi mẫu \((x_t, y_t)\) đến → dự đoán \(p_t\) → tính gradient → cập nhật trạng thái → trọng số \(w\) được suy ra từ trạng thái (không cập nhật \(w\) trực tiếp).

---

## 2. Trạng thái của thuật toán

FTRL không lưu trực tiếp **trọng số** \(w_i\). Với mỗi feature (bucket) \(i\), nó lưu:

- **\(z_i\):** tổng gradient đã điều chỉnh (adjusted gradient sum).
- **\(n_i\):** tổng bình phương gradient (sum of squared gradients).

Trọng số \(w_i\) được **tính khi cần** (lazy) từ \(z_i, n_i\) và các tham số \(\alpha, \beta, \lambda_1, \lambda_2\). Nhờ đó L1 regularization được áp dụng hiệu quả → nhiều \(w_i = 0\) (sparsity).

---

## 3. Vòng lặp xử lý một mẫu (Online Learning Loop)

Với mỗi mẫu \((x_t, y_t)\) (vector thưa + nhãn):

### Bước 1: Dự đoán (Prediction)
- Với mỗi bucket \(i\) có trong \(x_t\), tính \(w_{t,i}\) từ \((z_i, n_i)\) (công thức ở mục 4).
- Tính điểm: \(\text{score} = \sum_i w_{t,i} \cdot x_{t,i}\).
- Xác suất: \(p_t = \sigma(\text{score}) = \frac{1}{1 + e^{-\text{score}}}\).

### Bước 2: Tính gradient
- Sai số: \(g_t = p_t - y_t\).
- Gradient theo từng feature: \(g_{t,i} = g_t \cdot x_{t,i}\).

### Bước 3: Cập nhật \(n_i\)
\[
n_i \leftarrow n_i + g_{t,i}^2
\]
→ Feature xuất hiện nhiều / gradient lớn thì \(n_i\) tăng → learning rate (per-coordinate) giảm dần.

### Bước 4: Tính \(\sigma_i\)
\[
\sigma_i = \frac{\sqrt{n_i^{\text{mới}}} - \sqrt{n_i^{\text{cũ}}}}{\alpha}
\]
→ Điều chỉnh mức độ cập nhật theo lịch sử học của feature \(i\).

### Bước 5: Cập nhật \(z_i\)
\[
z_i \leftarrow z_i + g_{t,i} - \sigma_i w_{t,i}
\]
→ Tích lũy gradient mới và hiệu chỉnh theo trọng số hiện tại (gắn với regularization).

**Chỉ các feature có trong vector thưa \(x_t\)** mới được cập nhật; chi phí mỗi mẫu là O(số phần tử khác 0).

---

## 4. Công thức tính trọng số (Closed-form)

Khi cần \(w_i\) (bước dự đoán), FTRL dùng:

- **Nếu \(|z_i| \leq \lambda_1\):** \(w_i = 0\) → **sparsity** (feature bị “tắt”).
- **Ngược lại:**
  \[
  w_i = -\frac{z_i - \text{sgn}(z_i) \lambda_1}{(\beta + \sqrt{n_i})/\alpha + \lambda_2}
  \]

**Ý nghĩa:**
- **L1 (\(\lambda_1\)):** Ngưỡng để đưa \(w_i\) về 0.
- **Per-coordinate learning rate:** \((\beta + \sqrt{n_i})/\alpha\) — mỗi feature có tốc độ học riêng.
- **L2 (\(\lambda_2\)):** Kéo trọng số về gần 0, tránh overfitting.

---

## 5. Tham số FTRL-Proximal

| Tham số | Ý nghĩa | Mặc định dự án |
|--------|--------|-----------------|
| \(\alpha\) | Learning rate cơ bản | 0.1 |
| \(\beta\) | Làm mượt learning rate giai đoạn đầu | 1.0 |
| \(\lambda_1\) (L1) | Ngưỡng sparsity; càng lớn mô hình càng thưa | 1.0 |
| \(\lambda_2\) (L2) | Cường độ regularization L2 | 1.0 |

---

## 6. Tích hợp vào pipeline (StreamingTrainer)

- **StreamingTrainer** đọc từng mẫu từ StreamingIterator (đã qua tiền xử lý) → nhận vector thưa + label.
- Mỗi mẫu: gọi **`model.update(features, label)`** → bên trong thực hiện đúng vòng lặp 5 bước trên (dự đoán → gradient → cập nhật \(n, z\)).
- Sau `update`, metrics (log-loss, accuracy) được cập nhật; định kỳ in ra sparsity (số trọng số khác 0 / tổng số feature đã gặp).

→ **Giai đoạn 3** chính là bước “Predict & Update” trong vòng lặp đó: **áp dụng thuật toán tối ưu FTRL-Proximal** lên từng vector thưa, cập nhật mô hình ngay khi có dữ liệu mới và tạo ra mô hình thưa.

---

## 7. Kết quả của giai đoạn 3

- **Online learning:** Mô hình cập nhật ngay mỗi mẫu, không cần batch lớn.
- **Mô hình thưa:** Nhiều \(w_i = 0\) nhờ L1 → ít tham số lưu trữ, dự đoán nhanh.
- **Per-coordinate learning rate:** Học hiệu quả hơn khi feature có tần suất khác nhau.
- **Lưu trữ:** Trạng thái \((z, n)\) + hyperparameters trong file `.pkl` để dự đoán hoặc học tiếp (incremental learning).

---

## 8. Kết quả cuối cùng sau khi áp dụng FTRL-Proximal

Sau khi chạy xong giai đoạn 3 (train trên toàn bộ luồng dữ liệu hoặc đến khi dừng), ta có:

### 8.1. Mô hình đã học (trạng thái và trọng số)

| Thành phần | Mô tả |
|------------|--------|
| **Trạng thái \((z, n)\)** | Với mỗi feature (bucket) từng xuất hiện: \(z_i\) (tổng gradient đã điều chỉnh), \(n_i\) (tổng bình phương gradient). Đây là **đầu ra thực sự** mà FTRL lưu. |
| **Trọng số \(w\) (suy ra)** | Tính lazy từ \((z, n)\): nếu \(|z_i| \leq \lambda_1\) thì \(w_i = 0\); ngược lại \(w_i\) theo công thức FTRL. **Mô hình thưa**: chỉ một phần bucket có \(w_i \neq 0\). |
| **Hyperparameters** | \(\alpha, \beta, \lambda_1, \lambda_2\) đã dùng khi train — cần giữ nguyên khi load để dự đoán đúng. |

### 8.2. File mô hình (`.pkl`)

- **Nội dung lưu:** `z` (dict), `n` (dict), `alpha`, `beta`, `L1`, `L2`, `num_updates`.
- **Công dụng:** (1) **Dự đoán:** load → tính \(w\) từ \(z, n\) → với vector thưa mới, \(p = \sigma(w \cdot x)\). (2) **Học tiếp:** load → gọi `update` thêm trên dữ liệu mới (incremental learning).

#### Dùng file `.pkl` trong thực tế như thế nào?

**So với LLM:** LLM nhận **text** → **sinh text**. Mô hình FTRL trong file `.pkl` nhận **vector đặc trưng** (đã hash) → **trả về một số** trong \([0, 1]\): **xác suất dự đoán click** (CTR).

**Input khi dự đoán:** Một **vector thưa** (dict bucket → value) mô tả ngữ cảnh: ví dụ user, quảng cáo, thiết bị, giờ, v.v. — đã qua **cùng pipeline tiền xử lý** (Feature Hashing, cùng `num_buckets`) như lúc train.

**Output:** Một **số thực** \(p \in [0, 1]\). Ý nghĩa: "Xác suất (ước lượng) người dùng sẽ click trong ngữ cảnh này."

**Quy trình thực tế (inference):**
1. Load mô hình: `model = FTRLProximal.load("models/ftrl.pkl")`.
2. Có một **request** (ví dụ: user X xem ad Y lúc 10h trên mobile).
3. Tạo feature (ví dụ user_id, ad_id, hour, device, ...) → đưa qua **Preprocessor** (cùng cấu hình hash như train) → ra **vector thưa** `features`.
4. Gọi `p = model.predict(features)`.
5. **Dùng \(p\) trong nghiệp vụ:** xếp hạng quảng cáo (ad có \(p\) cao hơn ưu tiên), đấu thầu (bid theo \(p\)), lọc (bỏ ad có \(p\) quá thấp), A/B test, v.v.

**Học tiếp (incremental):** Load `.pkl` → đọc luồng dữ liệu mới (click/no-click) → với mỗi mẫu gọi `model.update(features, label)` → định kỳ save lại `.pkl`. Mô hình cập nhật theo dữ liệu mới mà không cần train lại từ đầu.

**Trong code dự án:** `python main.py --evaluate --model models/ftrl.pkl --data data/test.txt` sẽ load `.pkl`, đọc từng mẫu trong file test, gọi `model.predict(features)` (không update), rồi tính Log-Loss, Accuracy, AUC.

### 8.3. Đánh giá (metrics)

- **Log-Loss, Accuracy, AUC** (trên tập train/test) — đã tích lũy trong quá trình train/evaluate.
- **Sparsity:** tỉ lệ trọng số bằng 0 (số bucket có \(w_i \neq 0\) / tổng số bucket đã gặp). Ví dụ: "85% sparsity" nghĩa là 85% feature có trọng số 0.

### 8.4. Ví dụ: 10 triệu mẫu → bao nhiêu params? File .pkl chứa gì?

- **Số params không bằng số mẫu.** FTRL lưu trạng thái **theo từng feature (bucket)**, không theo từng mẫu. Số "params" = số **bucket duy nhất** đã từng xuất hiện và được cập nhật trong toàn bộ luồng dữ liệu.
- **1 bucket = 1 tham số** (một trọng số \(w_i\)). Công thức: `số params = len(z) = len(n)` = số bucket đã từng được cập nhật. Trong file .pkl mỗi bucket lưu 2 số thực \(z_i\), \(n_i\) (để suy ra \(w_i\) và cập nhật), không lưu \(w_i\) trực tiếp.
- **Với 10 triệu mẫu:** mỗi mẫu thường chỉ có vài chục đến vài trăm feature khác 0; do hash và sự lặp lại giữa các mẫu, số bucket **duy nhất** thường **nhỏ hơn nhiều** so với 10M. Ước lượng thực tế: từ **vài trăm nghìn đến vài triệu** bucket (tùy dữ liệu và `num_buckets`). Ví dụ: ~500k bucket → ~500k "params" (trong .pkl là 500k cặp \(z_i, n_i\)).
- **File .pkl** lưu: `alpha`, `beta`, `L1`, `L2`, `z` (dict bucket → float), `n` (dict bucket → float), `num_updates`. Kích thước file ≈ (số bucket × overhead cho 2 float + key) + hyperparams; ví dụ 500k bucket có thể vào cỡ vài MB đến vài chục MB.

### 8.5. Tóm tắt một dòng

**Kết quả cuối cùng** = **mô hình FTRL** (trạng thái \(z, n\) + hyperparameters), có thể **lưu thành file `.pkl`**, dùng để **dự đoán xác suất click** cho vector thưa mới hoặc **tiếp tục học**; mô hình **thưa** (nhiều \(w_i = 0\)) nên nhẹ và dự đoán nhanh.

---

## Tóm tắt một dòng

**Giai đoạn 3** nhận vector thưa và nhãn từ giai đoạn 2, **áp dụng thuật toán tối ưu FTRL-Proximal**: mỗi mẫu → dự đoán \(p\), tính gradient, cập nhật \((z_i, n_i)\) chỉ cho feature có trong mẫu; trọng số \(w_i\) tính từ \(z, n\) với L1/L2 → mô hình thưa, cập nhật ngay khi có dữ liệu mới.
