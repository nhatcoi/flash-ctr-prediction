# Chi tiết luồng Dữ liệu và Tiền xử lý (Data Flow & Preprocessing)

Đây là giải thích chi tiết về cách dữ liệu di chuyển từ tệp nén 1TB thành Vector toán học.

## 1. Trích xuất luồng (Data Loading Stream)

### Hàm thực thi: `CriteoDataLoader.__iter__`
- **Vị trí**: `src/data/data_loader.py` (Dòng 58-79)
- **Cấu chế**: Sử dụng `gzip.open` với mode `'rt'` để tạo một **text stream**.
- **Input**: Đường dẫn tệp `.gz` (Ví dụ: `data/day_2.gz`).
- **Xử lý nội bộ**: 
  - Sử dụng vòng lặp `for line in f` để trích xuất từng dòng văn bản thô. 
  - **Single-line Reading**: Tại mỗi thời điểm, chỉ có 1 `string` bản ghi tồn tại trong RAM.
  - Gọi hàm `_parse_line(line)`: Thực hiện `split('\t')`. 
- **Output**: Một bộ `(label, list_features)` (1 int, 39 strings).

---

## 2. Biến đổi dữ liệu (Feature Transformation)

### Hàm thực thi: `Preprocessor.transform()`
- **Vị trí**: `src/data/preprocessing.py`
- **Input**: Danh sách 39 chuỗi thô từ bước trên.
- **Quy trình 3 pha (3-Phase Pipeline)**:

#### PHA A: Xử lý giá trị trống (MissingValueHandler)
- **Code**: `self.missing_value_handler.handle(raw_features)`
- **Logic**: 
  - Nếu trường trống (empty string) trong 13 cột đầu $\rightarrow$ gán `-1`.
  - Nếu trường trống trong 26 cột sau $\rightarrow$ gán `"__MISSING__"`.

#### PHA B: Chuẩn hóa số học (Log Transformer)
- **Code**: `x_new = math.log(1 + x)`
- **Logic**: Áp dụng cho 13 biến integer (I1-I13). Điều này giải quyết hiện tượng "outliers" (giá trị quá lớn) làm Gradient bị nổ (exploding).

#### PHA C: Hashing Trick (Feature Hasher) - QUAN TRỌNG
- **Code**: `FeatureHasher.transform()`
- **Cơ chế**:
  1. Kết hợp tên cột và giá trị: `key = str(col_index) + "_" + str(val)`.
  2. Băm chuỗi `key` thành con số: `hash_val = zlib.adler32(key.encode())`.
  3. Ánh xạ vào bucket: `index = hash_val % (2**18)`.
- **Output**: Một Dictionary thưa kiểu `{index: 1.0}`.

---

## 3. Đầu ra của luồng Tiền xử lý
Sản phẩm cuối cùng của bước này là một **Sparse Vector**. 

- **Cấu trúc**: `{idx1: val1, idx2: val2, ...}`
- **Tính chất**: Dung lượng cực nhỏ (chỉ chứa các index xuất hiện), sẵn sàng nạp vào hàm `update` của thuật toán huấn luyện.

**Thông số kỹ thuật**:
- **Kích thước hash space**: $2^{18} = 262,144$ (Có thể cấu hình).
- **Phức tạp thời gian**: $O(F)$ với $F$ là số lượng đặc trưng (cố định = 39).
