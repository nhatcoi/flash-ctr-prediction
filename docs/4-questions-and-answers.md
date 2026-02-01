# Giải đáp thắc mắc và Deep Dive (Q&A and Deep Dive)

Tài liệu này tổng hợp các câu hỏi quan trọng phát sinh trong quá trình phát triển dự án và giải thích chuyên sâu về các lựa chọn kỹ thuật.

---

### 1. Tại sao huấn luyện trên Day 2 nhưng lại đánh giá trên Day 3?
Đây là một quy trình bắt buộc trong Machine Learning thực tế vì 3 lý do:
- **Tránh "Học vẹt" (Overfitting):** Nếu đánh giá trên chính dữ liệu đã học, mô hình sẽ có điểm rất cao nhưng thực chất chỉ là "ghi nhớ" đáp án. Dữ liệu Day 3 là hoàn toàn mới đối với mô hình.
- **Mô phỏng thực tế:** Hệ thống quảng cáo dùng dữ liệu hôm qua để dự đoán cho hôm nay. Việc này kiểm tra xem mô hình có thực sự hiểu quy luật để dự báo tương lai hay không.
- **Concept Drift:** Hành vi người dùng thay đổi theo ngày. Đánh giá trên Day 3 giúp kiểm tra tính bền bỉ của thuật toán trước sự thay đổi của thời gian.

### 2. Dự án có dùng hết 1TB dữ liệu không?
- **Khả năng:** Kiến trúc của chúng ta (Streaming + Hashing Trick) **đủ năng lực xử lý trọn vẹn 1TB** trên một máy tính cá nhân vì lượng RAM tiêu thụ là cố định ($O(1)$ memory).
- **Thực tế triển khai:** Trong phạm vi bài tập, chúng ta huấn luyện trên tập mẫu (1-10 triệu dòng) để chứng minh tính hội tụ. Vì tốc độ xử lý là tuần tự, việc chạy hết 4 tỷ dòng (~1TB) chỉ là vấn đề thời gian (mất khoảng 1-2 ngày), không phải vấn đề về thuật toán hay bộ nhớ.

### 3. Tại sao lại dùng thuật toán FTRL thay vì Spark?
- **FTRL (Follow-the-Regularized-Leader):** Cực kỳ mạnh trong việc tạo ra **mô hình thưa**. Trong dự đoán CTR, hàng triệu đặc trưng có thể xuất hiện nhưng chỉ có rất ít cái quan trọng. FTRL giúp loại bỏ nhiễu và làm file mô hình cực nhẹ.
- **Tại sao không dùng Spark?** Spark mạnh về xử lý song song (Parallel) nhưng gây ra "overhead" lớn khi phải đồng bộ trọng số giữa các máy. Với FTRL, việc cập nhật tuần tự (Streaming) trên Python thuần túy mang lại hiệu năng cao hơn và tiết kiệm tài nguyên hơn cho cấu hình máy đơn.

### 4. Bên trong file `.pkl` chứa dữ liệu gì?
File mô hình `.pkl` thực chất là một "Dictionary" chứa:
- **`z`**: Tổng tích lũy các Gradient.
- **`n`**: Tổng tích lũy bình phương các Gradient (dùng để điều chỉnh tốc độ học riêng cho từng đặc trưng).
- **Trọng số thưa**: Chỉ lưu các vị trí (index) có giá trị khác 0.
- **Metadata**: Các siêu tham số (alpha, beta, L1, L2) và số lượng dòng đã học.

### 5. Tại sao FTRL dùng để `--train` còn OLR dùng để `--compare`?
- **FTRL** là thuật toán tối ưu nhất cho bài toán này (đúng trọng tâm Đề 16).
- **OLR (Online Logistic Regression)** đóng vai trò là "Kẻ đối chứng" (Baseline). Chúng ta cần OLR để so sánh và chứng minh rằng: *"Dù cùng độ chính xác, nhưng FTRL cho mô hình nhẹ hơn (thưa hơn) rất nhiều so với OLR"*.

### 6. Đầu ra của mỗi dòng dữ liệu (Per-line Output) là gì?
Mỗi khi một dòng dữ liệu đi qua:
- **Input**: 1 dòng log nén.
- **Output tức thời**: Một con số xác suất (ví dụ: 0.12).
- **Hành động**: Tính sai số với nhãn thật (Gradient), sau đó cập nhật ngay lập tức vào bảng `z` và `n` trong RAM. Dòng dữ liệu đó sau đó bị xóa hoàn toàn khỏi RAM để nhường chỗ cho dòng tiếp theo.
