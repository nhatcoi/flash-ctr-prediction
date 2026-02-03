# Gợi ý trình bày slide: 2. Trạng thái của thuật toán

## Nội dung slide (đã có)

- **Tiêu đề:** 2. Trạng thái của thuật toán
- FTRL **không lưu trực tiếp** trọng số \(w_i\)
- Với mỗi feature (bucket) \(i\) lưu hai giá trị:
  - **\(z_i\):** tổng gradient đã điều chỉnh (adjusted gradient sum)
  - **\(n_i\):** tổng bình phương gradient (sum of squared gradients)
- Trọng số **\(w_i\) tính khi cần** (lazy) từ \(z_i, n_i\) và tham số \(\alpha, \beta, \lambda_1, \lambda_2\)
- **Lợi ích:** L1 regularization được áp dụng hiệu quả → nhiều \(w_i = 0\) (sparsity)

---

## Gợi ý trình bày (script ngắn)

### Mở đầu
- "Khác với SGD — lưu trực tiếp vector trọng số **w** — FTRL **không lưu w**. Nó lưu một **trạng thái nội bộ** cho từng feature."

### Hai biến trạng thái
- "Với **mỗi feature** (mỗi bucket sau hashing), FTRL chỉ lưu **hai số**: **z_i** và **n_i**."
- "**z_i** là tổng gradient đã được điều chỉnh theo lịch sử; **n_i** là tổng bình phương gradient — dùng để điều chỉnh learning rate riêng cho từng feature."

### Lazy weight
- "**Trọng số w_i không được cập nhật trực tiếp.** Khi cần (ví dụ lúc dự đoán), ta **tính w_i từ z_i, n_i** và các tham số alpha, beta, lambda1, lambda2 theo một công thức cố định. Đó là tính **lazy** — tính khi cần."

### Lợi ích / Sparsity
- "Nhờ cách này, **L1 regularization** được áp dụng rất hiệu quả: trong công thức tính w_i, nếu |z_i| nhỏ hơn ngưỡng lambda1 thì **w_i = 0**. Kết quả là **mô hình thưa** — rất nhiều trọng số bằng 0, tiết kiệm bộ nhớ và dự đoán nhanh."

### Chuyển slide
- "Slide tiếp theo sẽ cho thấy **công thức cụ thể** tính w_i và **năm bước cập nhật** z, n khi có mẫu mới."

---

## Điểm nhấn khi nói

1. **So sánh với SGD:** "SGD lưu w và cập nhật w trực tiếp; FTRL lưu (z, n) và suy w khi cần."
2. **Vì sao (z, n)?** "z tích lũy thông tin gradient có điều chỉnh; n tích lũy 'đã học bao nhiêu' để mỗi feature có learning rate riêng."
3. **Sparsity:** "Nhiều w_i = 0 không phải do ta ép, mà do công thức giải ra — |z_i| ≤ λ₁ thì w_i = 0."

---

## Có thể thêm trên slide (tùy chọn)

- **Sơ đồ nhỏ:**  
  `[z_i, n_i]` → (công thức) → `w_i` (chỉ khi |z_i| > λ₁)
- **Bảng tóm tắt:**

| Lưu trữ | Ý nghĩa |
|---------|----------|
| \(z_i\) | Tổng gradient đã điều chỉnh |
| \(n_i\) | Tổng bình phương gradient |
| \(w_i\) | Tính từ z_i, n_i khi cần (lazy) |

- **Một dòng takeaway:** "Trạng thái = (z, n) từng feature; w suy ra lazy → L1 hiệu quả → sparsity."

---

## Thời lượng

- Khoảng **1 phút**: nêu không lưu w → hai biến z, n → lazy w → lợi ích sparsity.

---

## Q&A nhanh

- **Tại sao không lưu w?** → Để khi tính w có bước "cắt" theo L1 (|z_i| ≤ λ₁ → w_i = 0); nếu cập nhật w trực tiếp như SGD thì L1 khó tạo đúng 0.
- **n_i dùng để làm gì?** → Trong mẫu số công thức w_i có \((\beta + \sqrt{n_i})/\alpha\) — learning rate riêng cho feature i, giảm dần khi n_i lớn (đã học nhiều).
- **Sparsity là gì?** → Phần lớn trọng số bằng 0; chỉ lưu và dùng những w_i ≠ 0 → mô hình nhẹ, nhanh.
