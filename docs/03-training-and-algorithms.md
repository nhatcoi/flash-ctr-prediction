# Thuật toán và Quy trình huấn luyện (Algorithms & Training)

Chương này giải thích cách `StreamingTrainer` điều phối việc huấn luyện và cơ chế toán học bên trong thuật toán FTRL.

## 1. Bộ điều phối trực tuyến (StreamingTrainer)

### Hàm thực thi: `StreamingTrainer.train()`
- **Vị trí**: `src/training/trainer.py` (Dòng 66-138)
- **Chế độ hoạt động**: Hoạt động theo vòng lặp **Input-Process-Update**.
- **Input**: Stream của các Vector thưa từ bộ Preprocessor.
- **Vòng lặp chi tiết**:
  1. Nhận Vector thưa (dict).
  2. Gọi `self.model.update(features, label)`.
  3. Nhận xác suất dự đoán `pred` trả về từ mô hình.
  4. Gửi `(label, pred)` vào `RunningMetrics` để cập nhật bảng log.
  5. Nếu đạt `log_interval`, in trạng thái (Loss, Acc, Sparsity) ra Console.

---

## 2. Thuật toán FTRL-Proximal (Cốt lõi)

### Hàm thực thi: `FTRLProximal.update()`
- **Vị trí**: `src/algorithms/ftrl.py` (Dòng 185-227)
- **Logic cập nhật từng bước (Per-line Update)**:

1.  **Dự đoán (Predict)**: Tính điểm $w \cdot x$ bằng cách chỉ lặp qua các `index` có trong vector thưa hiện tại.
    - $P = \sigma(\sum_{i \in \text{features}} w_i \cdot x_i)$
2.  **Tính Gradient**: $g = P - \text{label}$.
3.  **Cập nhật biến trạng thái (Internal State)**:
    - Với mỗi $i \in \text{features}$:
      - $g_i = g \cdot x_i$
      - $\sigma_i = \frac{\sqrt{n_i + g_i^2} - \sqrt{n_i}}{\alpha}$
      - $z_i = z_i + g_i - \sigma_i \cdot w_i$
      - $n_i = n_i + g_i^2$
4.  **Cơ chế Sparsity (Proximal L1)**: 
    - Trọng số $w_i$ thực tế chỉ được tính khi $|z_i| > \lambda_1$. Nếu không, $w_i = 0$.

---

## 3. Cấu trúc mô hình đầu ra (.pkl)

### Hàm thực thi: `FTRLProximal.save()`
Khi kết thúc huấn luyện, mô hình được đóng gói thành file binary.

**Nội dung bên trong file `.pkl`**:
1.  **`z` (Gradient sum)**: Dictionary lưu trữ các giá trị $z$ cho từng index băm.
2.  **`n` (Squared gradient sum)**: Dictionary lưu trữ các giá trị $n$ (dùng cho Adaptive Learning Rate).
3.  **Hyperparameters**: Lưu $\alpha, \beta, \lambda_1, \lambda_2$ để đảm bảo tính nhất quán khi dự đoán.
4.  **Metadata**: `num_updates` (tổng số dòng đã học).

**Lợi ích**: File này chứa "trạng thái huấn luyện", cho phép chúng ta tiếp tục học (Incremental learning) từ đúng vị trí đã dừng lại mà không cần train lại từ dòng 0.
