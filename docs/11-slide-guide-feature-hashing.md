# Slide: Feature Hashing (Hashing Trick)

Trình bày tương tự bước Log Transform — có input, output, công thức và ví dụ cụ thể.

---

## Tiêu đề & Vị trí trong pipeline

- **Tiêu đề:** Feature Hashing (Hashing Trick)
- **Luồng:** Missing Values → Log Transform → **Feature Hashing** → Vector thưa

---

## Định nghĩa

Feature Hashing là bước **băm mỗi đặc trưng (tên + giá trị)** thành **một chỉ số bucket** trong không gian cố định \(N\) (ví dụ \(N = 2^{18}\)), rồi ghi nhận giá trị (và dấu ±1 nếu dùng signed hashing) vào vector thưa. Áp dụng cho **cả 13 integer (đã log)** và **26 categorical**.

---

## Công thức

- **Chuẩn:**  
  \(h(\text{feature}) = \text{hash}\big(\text{name} + \texttt{":"} + \text{value}\big) \bmod N\)
- **Signed hashing:** thêm dấu \(\pm 1\) từ hash phụ để giảm bias khi collision.
- **Kết quả:** mỗi cặp `(name, value)` → một **bucket index** trong \([0, N-1]\) và một **value** (số thực, có thể âm nếu sign = -1).

---

## Mục đích

- **Nén không gian:** Categorical có cardinality rất cao (hàng triệu giá trị) → one-hot không khả thi. Hashing đưa mọi feature vào **N bucket cố định**.
- **Bộ nhớ cố định:** Chỉ cần \(N\) trọng số, không phụ thuộc số giá trị unique.
- **O(1) tra cứu:** Không cần dictionary; chỉ cần hash và mod.
- **Unseen features:** Feature chưa từng gặp vẫn được hash vào một bucket.

---

## Input (đầu vào của bước Feature Hashing)

Đầu vào là **sau** bước Missing Values và Log Transform — tức là **cùng một mẫu** đã được xử lý:

- **13 integer features:** giá trị đã xử lý thiếu (-1 hoặc bỏ qua) và **đã log** \(y = \ln(1+x)\) (ví dụ: 0.693, 0, 7.232, …). Các vị trí missing (-1) thường đã bị bỏ qua, không đưa vào hashing.
- **26 categorical features:** chuỗi (ví dụ "68fd1e64", "88e26c9b") hoặc token missing (ví dụ "__MISSING__"). Giá trị rỗng '' có thể đã được thay bằng __MISSING__ hoặc bỏ qua.

**Dạng trong code:** list 39 phần tử (13 số + 26 chuỗi), ví dụ:

```text
[0.693, 0, 1.792, 0, (bỏ qua), 7.232, 1.386, ... , "68fd1e64", "88e26c9b", "fb936136", ...]
```

Hoặc vẫn là `raw_features` gốc nhưng **bên trong** preprocessor đã gọi Log Transform trước, rồi mới đưa từng (name, value) vào hashing — khi đó **input thực sự của bước hash** là các cặp (tên feature, giá trị đã xử lý).

---

## Output (đầu ra của bước Feature Hashing)

- **Kiểu:** `Dict[int, float]` — từ điển thưa (chỉ lưu các bucket được dùng).
- **Key:** bucket index (số nguyên trong \([0, N-1]\)), ví dụ 218608, 153521, 50667.
- **Value:** giá trị feature sau khi xử lý:
  - **Bias:** một bucket cố định (vd. 218608) có value **1.0**.
  - **Integer (đã log):** value = giá trị log × dấu (±1), ví dụ 0.693, -1.792, 7.232.
  - **Categorical:** thường **1.0** hoặc **-1.0** (signed hashing).
- **Collision:** nhiều feature hash cùng bucket thì value **cộng dồn** (cộng/trừ theo dấu).

**Ví dụ output:**

```text
{
  218608: 1.0,    # Bias
  153521: 0.693,  # I1: ln(1+1), sign +1
  195592: 0.0,    # I2: 0
  50667: 7.232,   # I6: ln(1+1382)
  175177: 1.792,  # I3: ln(1+5)
  ...,
  89201: -1.0,    # C1 categorical, sign -1
  200441: 1.0     # C2 categorical
}
```

---

## Xử lý giá trị đặc biệt

- **Missing (integer -1):** bước Log Transform đã bỏ qua → **không** tạo cặp (name, value) cho bước hash → không có bucket nào cho feature đó trong output.
- **Missing (categorical ''/__MISSING__):** có thể (a) bỏ qua, hoặc (b) hash với value "__MISSING__" → vẫn cho một bucket, mô hình học được pattern “missing”.
- **Collision:** nhiều (name, value) cùng bucket → value **cộng dồn**; signed hashing giúp giảm bias khi cộng.

---

## Cách tính một số value cụ thể (ví dụ như slide Log Transform)

| Value trong output | Nguồn |
|--------------------|--------|
| **1.0 (Bias)** | Bucket từ hash("__BIAS__"), value cố định 1.0. |
| **0.693** | Integer index 0 = 1 → \(\ln(1+1) \approx 0.693\), hash với name "I1", sign +1 → bucket 153521. |
| **0.0** | Integer = 0 → \(\ln(1+0)=0\) → value 0.0. |
| **7.232** | Integer index 5 = 1382 → \(\ln(1+1382) \approx 7.232\), hash "I6" → bucket 50667. |
| **1.0 / -1.0 (Categorical)** | Chuỗi ví dụ "68fd1e64" → hash("C1:68fd1e64") mod N → bucket; value = +1 hoặc -1 theo signed hashing. |

---

## Gợi ý minh họa bên cạnh slide (như ảnh Log Transform)

**Bên trái (chữ):** Định nghĩa, công thức, mục đích, Input/Output, bảng value như trên.

**Bên phải (hình):** Giữ cùng phong cách “debugger / data flow”:

1. **Input:** Một list hoặc “sau Log Transform”:
   - 13 số (0.693, 0, 1.792, 0, …, 7.232, …) + 26 chuỗi ("68fd1e64", "88e26c9b", …).
2. **Khối giữa:** Hàm/ô “Feature Hashing” với công thức \(h = \text{hash}(\text{name}+\texttt{:}+\text{value}) \bmod N\).
3. **Output:** Một `dict` với các cặp bucket : value, ví dụ:
   - 218608: 1.0  
   - 153521: 0.693  
   - 195592: 0.0  
   - 50667: 7.232  
   - 89201: -1.0  
   - …

Có thể thêm mũi tên: vài dòng “I1:0.693 → bucket 153521”, “C1:68fd1e64 → bucket 89201” để rõ input từng feature → bucket và value trong output.

---

## Tóm tắt một dòng

- **Input:** Cùng mẫu đã qua Missing + Log: 13 số (đã log) + 26 categorical (chuỗi).
- **Output:** `Dict[int, float]` — key = bucket index, value = 1.0 (bias), giá trị log×sign (integer), hoặc ±1.0 (categorical).

Nếu bạn gửi thêm ảnh slide Log Transform (file hoặc link), có thể chỉnh lại cho khung hình và font cho giống hệt (tiêu đề, màu, vị trí input/output).
