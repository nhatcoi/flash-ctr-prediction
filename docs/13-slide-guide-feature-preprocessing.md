# Slide gợi ý: 3.4. Tiền xử lý đặc trưng (Feature Preprocessing)

Gợi ý chia nội dung thành từng slide, kèm mô tả ảnh/minh họa đặt bên cạnh.

---

## Slide 1: Tiêu đề section

**Nội dung slide (giữa hoặc trái):**
- **Tiền xử lý đặc trưng (Feature Preprocessing)**
- 3 bước: Missing Values → Log Transform → Feature Hashing

**Ảnh/minh họa gợi ý (bên phải):**
- Sơ đồ pipeline 3 ô nối tiếp: `Raw → Missing → Log → Hash → Sparse`
- Hoặc ảnh minh họa “ống dẫn dữ liệu” từ dạng list 39 phần tử sang dict thưa.

---

## Slide 2: Xử lý giá trị thiếu (Missing Values)

**Nội dung slide (trái, bullet ngắn):**

- **Integer features**
  - Thay thiếu → **-1** (marker)
- **Categorical features**
  - Giữ `''` hoặc token **`__MISSING__`**
- **Lý do**
  - Online learning: không tính mean/median trước được
  - Marker cho mô hình học pattern “missing” như một feature riêng

**Ảnh/minh họa gợi ý (bên phải):**
- Bảng 2 cột: **Trước** (ô trống, `''`) | **Sau** (`-1`, `__MISSING__`)
- Hoặc 1 dòng TSV với ô trống → mũi tên → cùng dòng với -1 và __MISSING__ điền vào ô trống.

---

## Slide 3: Log Transformation cho đặc trưng số

**Nội dung slide (trái):**

- **Công thức:** \( x' = \log(1 + x) \), \( x \geq 0 \)
- **Lý do**
  - Criteo: phân phối **lệch phải** (nhiều giá trị nhỏ, ít giá trị rất lớn)
  - Log → phân phối gần chuẩn hơn → mô hình học tốt hơn
- **Áp dụng:** 13 integer features

**Ảnh/minh họa gợi ý (bên phải):**
- **Histogram** 2 hình cạnh nhau:
  - Trái: phân phối lệch phải (đuôi dài bên phải)
  - Phải: sau log, hình chuông hơn
- Hoặc đồ thị \( y = \log(1+x) \) với trục x từ 0 đến giá trị lớn.

---

## Slide 4: Feature Hashing – Vấn đề

**Nội dung slide (trái):**

- **Categorical:** cardinality rất cao (hàng triệu giá trị unique)
- **One-hot** → không gian đặc trưng quá lớn, không khả thi
- **Cần:** biểu diễn trong không gian **cố định**, bộ nhớ hữu hạn

**Ảnh/minh họa gợi ý (bên phải):**
- Sơ đồ “one-hot explosion”: vài feature → ma trận khổng lồ (nhiều cột)
- Hoặc biểu đồ: trục ngang = số feature unique, trục dọc = “kích thước không gian” tăng vọt.

---

## Slide 5: Feature Hashing – Công thức và cơ chế

**Nội dung slide (trái):**

- **Công thức:**
  \[
  h(\text{feature}) = \text{hash}(\text{name} + \texttt{":"} + \text{value}) \bmod N
  \]
- **N:** số bucket (vd. \( 2^{18} = 262{,}144 \))
- Mỗi (name, value) → **một** bucket index trong \([0, N-1]\)

**Ảnh/minh họa gợi ý (bên phải):**
- Sơ đồ: vài cặp `(I1:100)`, `(C2:68fd1e64)` → mũi tên vào **hàm hash** → ra các số (vd. 12453, 89201) nằm trong N bucket.
- Có thể vẽ N ô (bucket) và vài mũi tên từ feature vào ô.

---

## Slide 6: Feature Hashing – Ưu điểm & Signed Hashing

**Nội dung slide (trái):**

- **Ưu điểm**
  - Bộ nhớ **cố định** (N trọng số)
  - **O(1)** lookup, không dictionary
  - **Unseen features** vẫn hash được
- **Signed Hashing**
  - Gán \(+1\) hoặc \(-1\) theo hash phụ
  - Giảm **bias do collision**

**Ảnh/minh họa gợi ý (bên phải):**
- Hình: 2 feature khác nhau hash cùng bucket → một có dấu +1, một -1 → khi cộng giá trị triệt tiêu bớt (minh họa giảm bias).
- Hoặc bảng nhỏ: Feature A → bucket 5, +1; Feature B → bucket 5, -1.

---

## Slide 7: Kết quả – Vector thưa

**Nội dung slide (trái):**

- **Output:** `Dict[int, float]` (vector thưa)
  - **Key:** bucket index (0 → N-1)
  - **Value:** giá trị feature (tích lũy nếu collision)
- Chỉ lưu **index có giá trị ≠ 0** → thưa, tiết kiệm bộ nhớ

**Ảnh/minh họa gợi ý (bên phải):**
- Ví dụ dict: `{12453: 0.69, 89201: -1.0, 200441: 1.0, ..., 187262: 1.0}` (bias)
- Hoặc minh họa vector dài toàn 0, chỉ vài vị trí có số (sparse vector).

---

## Gợi ý layout chung

- **Trái (2/3):** chữ, công thức, bullet.
- **Phải (1/3):** ảnh/sơ đồ/minh họa.
- Mỗi slide 1 ý chính; công thức đặt riêng một dòng cho dễ đọc.
- Có thể thêm slide “Tóm tắt 3 bước” cuối: 3 ô Raw → Missing → Log → Hash → Sparse, kèm 1 câu cho mỗi bước.

Nếu bạn gửi công cụ làm slide (PowerPoint, Google Slides, LaTeX Beamer, v.v.) mình có thể chuyển thành outline từng slide hoặc gợi ý text từng ô.
