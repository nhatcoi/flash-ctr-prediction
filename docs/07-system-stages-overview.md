Dưới đây là nội dung kỹ thuật của bạn đã được format lại để trình bày rõ ràng, mạch lạc và chuyên nghiệp hơn, phù hợp cho việc làm báo cáo hoặc tài liệu dự án.

---

## 🏗️ KIẾN TRÚC HỆ THỐNG XỬ LÝ DỮ LIỆU LỚN (1TB CRITEO DATASET)

### 🚀 Giai Đoạn 1: Data Loader (Streaming Architecture)

* **Vấn đề:** Dataset 1TB nén `.gz` (tương đương ~3-4TB giải nén) quá lớn để nạp vào RAM và quá tốn kém chi phí I/O/thời gian để giải nén ra ổ cứng.
* **Giải pháp:** Sử dụng cơ chế **Lazy Loading** (Stream-based reading).
* **Luận điểm kỹ thuật:** Xử lý dữ liệu với độ phức tạp không gian  cho mỗi bước đọc.
* **Cơ chế hoạt động:**
* Sử dụng `gzip.open` kết hợp với generator (`yield`).
* Thay vì dùng `f.read()` toàn bộ, hệ thống dùng `f.readline()` để nạp đúng một dòng duy nhất vào RAM tại một thời điểm.
* Sau khi dòng đó được xử lý, nó lập tức được giải phóng để nhường chỗ cho dòng tiếp theo.


* **🏁 Kết luận:** Hệ thống có thể xử lý file dữ liệu lớn vô hạn (vài TB hay PB) mà **không bao giờ bị lỗi "Out of Memory"** ở giai đoạn đọc.

---

### ⚙️ Giai Đoạn 2: Preprocessing & Hashing (Fixed-Memory Architecture)

* **Vấn đề:** Dữ liệu Criteo chứa hàng trăm triệu ID khác nhau (User ID, Ad ID...). Việc sử dụng Dictionary Mapping truyền thống sẽ khiến RAM phình to tỷ lệ thuận với số lượng ID mới (High Cardinality).
* **Giải pháp:** Sử dụng **Hashing Trick** (Feature Hashing).
* **Luận điểm kỹ thuật:** Ép không gian đặc trưng vô hạn về kích thước cố định .
* **Cơ chế hoạt động:**
* Mọi ID được đưa qua hàm băm và lấy dư cho  (thường là  hoặc ).
* ID mới xuất hiện sẽ tự động rơi vào một trong  ô có sẵn (chấp nhận va chạm nhỏ).
* Dù số lượng ID tăng lên 4 tỷ, bộ nhớ sử dụng vẫn chỉ tốn đúng  ô cố định.


* **🏁 Kết luận:** Giải quyết triệt để bài toán độ thưa (**Sparsity**) và số chiều lớn (**High Cardinality**), giữ cho bộ nhớ luôn là hằng số.

---

### 🧠 Giai Đoạn 3: FTRL Training (Online Learning)

* **Vấn đề:** Các thuật toán Batch Learning truyền thống yêu cầu toàn bộ dữ liệu phải sẵn sàng trong bộ nhớ để tính trọng số. Với 4 tỷ dòng dữ liệu, điều này là bất khả thi.
* **Giải pháp:** Thuật toán **FTRL-Proximal** (Follow-the-Regularized-Leader).
* **Luận điểm kỹ thuật:** Học trực tuyến (**Online Learning**) kết hợp khả năng tạo mô hình thưa (**Sparsity**).
* **Cơ chế hoạt động:**
* **Học cuốn chiếu:** Lấy 1 dòng  Tính Gradient  Cập nhật bảng tham số  và   Hủy dòng đó.
* **L1 Regularization:** Cơ chế cực mạnh giúp tự động ép trọng số của các ID "nhiễu/rác" về đúng bằng 0.


* **🏁 Kết luận:** Sau khi "quét" qua 1TB dữ liệu, kết quả thu được không phải là file nặng hàng TB, mà là một file model `.pkl` chỉ nặng vài MB, chứa đựng những trọng số tinh túy nhất.

---

## 🏆 TỔNG KẾT: CÂU TRẢ LỜI CHO BÀI TOÁN 1TB

Để trả lời câu hỏi *"Dự án này xử lý 1TB như thế nào?"*, đây là 3 trụ cột kỹ thuật chính:

1. **Về Bộ nhớ (Memory):** Hệ thống sử dụng kiến trúc **Fixed-Memory** (). RAM được giới hạn ở mức cố định, không bao giờ tăng theo dung lượng dữ liệu đầu vào.
2. **Về Thời gian (Time):** Hệ thống sử dụng kiến trúc **One-Pass** (). Dữ liệu chỉ cần chảy qua mô hình đúng 1 lần, không cần quay lại (re-read), tối ưu tuyệt đối cho luồng dữ liệu 4-5 tỷ dòng.
3. **Về Kết quả (Outcome):** Hệ thống biến 1TB dữ liệu thô hỗn độn thành một **"Bộ não"** (Model `.pkl`) siêu nhẹ và siêu thưa, có khả năng dự đoán xác suất Click chính xác trong vài mili giây.