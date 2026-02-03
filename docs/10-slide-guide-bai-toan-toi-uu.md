# Gợi ý trình bày slide: 1. Bài toán tối ưu

## Cấu trúc slide (đã có)

- **Tiêu đề:** 1. Bài toán tối ưu
- **Mục tiêu:** Dự đoán p = σ(w·x), tối thiểu hóa Log-Loss + regularization
- **Công thức:** min_w Σ_t ℓ_t(w) + λ₁‖w‖₁ + λ₂‖w‖₂²
- **Cách giải:** FTRL-Proximal online: mẫu → dự đoán → gradient → cập nhật trạng thái → w suy từ trạng thái

---

## Gợi ý trình bày (script ngắn)

### Mở đầu (1–2 câu)
- "Phần này nêu **bài toán tối ưu** mà hệ thống CTR cần giải: vừa dự đoán xác suất click chính xác, vừa kiểm soát độ phức tạp mô hình."

### Mục tiêu
- "**Mục tiêu** là dự đoán xác suất click **p** bằng mô hình logistic: **p = σ(w·x)**, với **x** là vector đặc trưng thưa sau Feature Hashing."
- "Chúng ta không chỉ tối thiểu hóa **Log-Loss** (sai số dự đoán) mà còn thêm **regularization** để mô hình ổn định và thưa."

### Công thức
- "Công thức tổng quát: cực tiểu theo **w** của **tổng Log-Loss theo từng mẫu**, cộng **L1** (λ₁ nhân chuẩn L1 của w) và **L2** (λ₂ nhân bình phương chuẩn L2)."
- "**L1** đẩy nhiều trọng số về 0 → mô hình thưa; **L2** hạn chế trọng số lớn → tránh overfitting."

### FTRL-Proximal
- "Bài toán này được giải **online** bằng thuật toán **FTRL-Proximal**."
- "Luồng xử lý: mỗi khi có mẫu **(x_t, y_t)** → mô hình **dự đoán p_t** → tính **gradient** → **cập nhật trạng thái nội bộ** (z, n); **trọng số w không cập nhật trực tiếp** mà được **suy ra từ trạng thái** khi cần. Đây là điểm khác biệt quan trọng so với SGD thuần."

### Kết nối sang slide sau
- "Slide tiếp theo sẽ đi vào **trạng thái (z, n)** và **công thức cập nhật** từng bước."

---

## Gợi ý bổ sung trên slide (tùy chọn)

1. **Chú thích ngắn dưới công thức**
   - Σ_t ℓ_t(w): tổng Log-Loss theo mẫu
   - λ₁‖w‖₁: L1 → sparsity
   - λ₂‖w‖₂²: L2 → tránh overfitting

2. **Một dòng "takeaway"**
   - "Bài toán: min Log-Loss + L1 + L2. Giải bằng FTRL-Proximal (online, cập nhật trạng thái, w suy từ trạng thái)."

3. **Hình minh họa nhỏ (nếu có chỗ)**
   - Sơ đồ luồng: (x_t, y_t) → Predict → Gradient → Update state (z,n) → w = f(z,n)

---

## Thời lượng gợi ý

- **1 slide:** khoảng 1–1,5 phút (mục tiêu + công thức + FTRL ngắn).
- Nếu tách: slide 1 = Mục tiêu + Công thức; slide 2 = FTRL-Proximal (online, trạng thái, w suy từ trạng thái).

---

## Câu hỏi thường gặp khi Q&A

- **Tại sao không cập nhật w trực tiếp?** → Để áp dụng L1 hiệu quả: FTRL lưu (z, n), khi tính w có bước "soft-threshold" theo λ₁ nên nhiều w_i = 0.
- **Online nghĩa là gì?** → Mỗi mẫu đến một lần, cập nhật ngay, không cần lưu toàn bộ dataset hay lặp nhiều epoch.
- **L1 và L2 khác nhau thế nào?** → L1 đẩy đúng 0 (thưa); L2 làm nhỏ trọng số nhưng hiếm khi đúng 0.
