# Tổng quan kỹ thuật dự án (Technical Project Overview)

Dự án này triển khai hệ thống dự đoán Click-Through Rate (CTR) quy mô lớn, tối ưu hóa cho bộ dữ liệu **Criteo 1TB**. Thay vì tiếp cận theo cách học máy truyền thống (Batch Learning), dự án sử dụng kiến thức về **Thuật toán ứng dụng** để xử lý dữ liệu dưới dạng luồng (Streaming) và cập nhật mô hình trực tuyến (Online Learning).

## 4. Kiến trúc hệ thống: Tại sao dùng Streaming Python thay vì Spark?

Trong bài toán xử lý 1TB dữ liệu, Apache Spark thường được nhắc đến như một lựa chọn mặc định. Tuy nhiên, dự án này lựa chọn kiến trúc **Lightweight Streaming Python** vì các lý do chiến lược sau:

### a. Sự phù hợp với thuật toán và bản chất dữ liệu
- **Online Learning (Cập nhật từng dòng)**: Thuật toán FTRL cập nhật trọng số dựa trên sai số của *từng mẫu dữ liệu* một cách tuần tự. Spark mạnh về xử lý song song trên nhiều node, nhưng việc đồng bộ hóa trọng số giữa các node trong Spark thường gây ra "overhead" (độ trễ truyền tin) rất lớn.
- **Duy trì trạng thái (Statefulness)**: FTRL cần duy trì biến `z` và `n` liên tục. Việc triển khai xử lý luồng tuần tự trên Python giúp quản lý trạng thái này một cách trực tiếp và cực kỳ nhanh.

### b. Tối ưu hóa tài nguyên (Efficiency vs Brute Force)
- **Hashing Trick**: Nhờ kỹ thuật băm đặc trưng, chúng ta cố định được bộ nhớ RAM sử dụng (khoảng 200-500MB). Điều này cho phép xử lý 1TB dữ liệu trên **duy nhất một máy tính cá nhân** mà không cần tới một cụm máy chủ (Cluster) đắt tiền chạy Spark.
- **Loại bỏ Data Shuffling**: Spark thường tốn nhiều tài nguyên cho việc xáo trộn dữ liệu (shuffling) giữa các máy. Kiến trúc của chúng ta đọc dữ liệu trực tiếp từ file nén và xử lý ngay lập tức, loại bỏ hoàn toàn lãng phí này.

### c. Khả năng mở rộng (Scalability)
Mặc dù dự án hiện tại chạy trên Node đơn (Single-Node), kiến trúc này hoàn toàn có thể mở rộng:
1. **Tiền xử lý**: Có thể dùng Spark để ETL dữ liệu thô (nếu cần thống kê tổng thể).
2. **Serving**: File mô hình `.pkl` cực nhẹ có thể được triển khai trên hàng ngàn Microservices để dự đoán song song với độ trễ Mili giây.

---
Tóm lại, việc sử dụng **Streaming Python** không phải là một hạn chế, mà là một **sự lựa chọn tối ưu** dựa trên các kiến thức về cấu trúc dữ liệu và giải thuật ứng dụng, giúp giải quyết bài toán Big Data với chi phí tài nguyên thấp nhất.

## 1. Định nghĩa toán học bài toán
Mục tiêu là tối ưu hóa hàm dự đoán $P(\text{click} | x)$, trong đó $x$ là vector đặc trưng đại diện cho người dùng và ngữ cảnh quảng cáo.

- **Objective Function**: Cực tiểu hóa hàm mất mát Logarithmic Loss (Log-Loss):
  $$L = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$
- **Constraints**: 
  - Độ trễ huấn luyện (Training Latency) $< 1 \text{ms/sample}$.
  - Bộ nhớ RAM tiêu thụ cố định, không phụ thuộc kích thước dữ liệu (cố định bởi số lượng hash buckets).

## 2. Các thành phần kỹ thuật cốt lõi (Core Components)

Hệ thống được xây dựng dựa trên 3 trụ cột kỹ thuật "trắng" (không sử dụng thư viện đen - black box):

1.  **Dữ liệu luồng (Data Streaming)**: Triển khai thông qua `StreamingIterator`, giải quyết bài toán đọc dữ liệu Terabyte mà không giải nén.
2.  **Mã hóa đặc trưng (Feature Engineering)**: Sử dụng **Hashing Trick** (Feature Hashing) để cố định không gian chiều đặc trưng, giải quyết bài toán Sparsity của Criteo.
3.  **Tối ưu hóa lồi trực tuyến (Online Optimization)**: Sử dụng thuật toán **FTRL-Proximal**, cho phép cập nhật trọng số mô hình ngay lập tức khi có dữ liệu mới và tạo ra mô hình thưa (Sparse model).

## 3. Quy trình thực thi Command (Step-by-Step Flow)

Khi bạn thực hiện lệnh `python main.py --train`, hệ thống sẽ vận hành theo luồng 5 bước khép kín sau:

1.  **Bước 1: Mở luồng cung cấp (Streaming Iterator)**
    - **Hàm**: `CriteoDataLoader.__iter__` phối hợp với `gzip.open`.
    - **Hoạt động**: Mở tệp `.gz`, trích xuất đúng 1 dòng văn bản thô. Không giải nén toàn bộ file để bảo vệ ổ cứng và RAM.
2.  **Bước 2: Chế biến đặc trưng (Preprocessing)**
    - **Hàm**: `Preprocessor.transform`.
    - **Hoạt động**: Điền giá trị thiếu (`-1`), chuẩn hóa Log cho số học, và thực hiện **Hashing Trick** để biến chuỗi thành các con số Index trong khoảng $0 \to 262,143$.
3.  **Bước 3: Dự đoán & Cập nhật (FTRL Update)**
    - **Hàm**: `FTRLProximal.update`.
    - **Hoạt động**: 
        - Dự đoán xác suất click hiện tại ($P$).
        - So sánh với nhãn thật ($P - y$) để tìm Gradient.
        - Cập nhật tức thì vào bảng trạng thái `z` và `n` trong RAM.
4.  **Bước 4: Theo dõi hiệu năng (Metrics Update)**
    - **Hàm**: `RunningMetrics.update`.
    - **Hoạt động**: Tính Log-loss và Accuracy của dòng vừa rồi, tích lũy vào trung bình cộng để hiển thị lên thanh tiến trình (Progress Bar).
5.  **Bước 5: Đóng gói trí tuệ (Save Model)**
    - **Hàm**: `FTRLProximal.save`.
    - **Hoạt động**: Sau khi chạy hết số dòng yêu cầu, toàn bộ bảng số `z, n` được lưu vào file `.pkl` để phục vụ cho việc dự đoán hoặc học tiếp sau này.

Tài liệu này sẽ đi sâu vào từng bước kỹ thuật ở các chương tiếp theo.
