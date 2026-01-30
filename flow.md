1. Phát biểu bài toán (Problem Formulation)
Thay vì chỉ là "chọn quảng cáo", bạn nên cụ thể hóa thành bài toán: "Tối đa hóa giá trị kỳ vọng của lượt click (Expected Click-Through Rate - eCTR) dưới ràng buộc về tài nguyên và độ trễ (latency)."

Input: Stream dữ liệu người dùng (context, user features) và danh mục quảng cáo (ad features).
Output: Danh sách $k$ quảng cáo có khả năng được click cao nhất.
2. Thách thức kỹ thuật (Challenges)
Data Scale: 1TB dữ liệu không thể nạp vào RAM.
Sparsity: Đặc trưng của Criteo chủ yếu là Categorical đã được hash, cực kỳ thưa (high cardinality).
Concept Drift: Sở thích người dùng thay đổi theo thời gian, mô hình cần cập nhật liên tục (Streaming).
3. Thuật toán đề xuất (Recommended Algorithms)
Để gây ấn tượng trong môn Thuật toán ứng dụng, bạn nên tập trung vào các nhóm thuật toán sau:

a. Thuật toán Học máy trực tuyến (Online Learning)
FTRL-Proximal (Follow-the-Regularized-Leader): Đây là "thuật toán vàng" trong dự đoán CTR của Google. Nó cực kỳ hiệu quả với dữ liệu thưa và lớn vì nó tạo ra các mô hình thưa (nhiều trọng số bằng 0), giúp tiết kiệm bộ nhớ và dự đoán nhanh.
Online Gradient Descent: Cập nhật trọng số ngay khi có một bản ghi mới (hoặc mini-batch).
b. Thuật toán Xử lý dữ liệu lớn (Streaming & Sketching)
Hashing Trick (Feature Hashing): Sử dụng hàm hash để nén không gian đặc trưng từ hàng triệu chiều xuống một không gian cố định (ví dụ $2^{20}$), giải quyết vấn đề bộ nhớ khi xử lý 1TB dữ liệu.
Reservoir Sampling: Nếu bạn cần lấy mẫu một tập con từ stream dữ liệu để huấn luyện mà không biết trước kích thước tập dữ liệu.
c. Thuật toán Tối ưu hóa hiển thị (Exploration vs. Exploitation)
Multi-Armed Bandits (UCB hoặc Thompson Sampling): Giúp chọn quảng cáo không chỉ dựa trên kết quả quá khứ (Exploitation) mà còn thử nghiệm các quảng cáo mới (Exploration) để tìm ra "ngôi sao" mới.
4. Kiến trúc hệ thống gợi ý (Tech Stack)
Do bạn làm bài tập lớn, có thể mô phỏng luồng streaming như sau:

Lưu trữ: Sử dụng định dạng Parquet hoặc Avro (nén tốt hơn CSV nhiều lần) để đọc dữ liệu từ ổ cứng.
Xử lý:
PySpark: Để xử lý batch lớn và trích xuất đặc trưng.
Spark Streaming: Để giả lập luồng dữ liệu 1TB chảy vào hệ thống.
Model: Tự cài đặt thuật toán FTRL (để thể hiện khả năng thuật toán) hoặc dùng thư viện Vowpal Wabbit (thư viện mạnh nhất thế giới cho online learning trên dữ liệu lớn).
5. Lộ trình triển khai (Roadmap)
Giai đoạn 1 (EDA & Preprocessing):
Phân tích cấu trúc 40 features (13 numerical, 26 categorical).
Xử lý missing values bằng log-transform (cho numerical) và Hashing Trick (cho categorical).
Giai đoạn 2 (Algorithm Implementation):
Cài đặt thuật toán FTRL-Proximal.
Thử nghiệm trên một mẫu nhỏ (ví dụ 1GB đầu tiên).
Giai đoạn 3 (Scaling):
Đưa vào pipeline Spark để xử lý trên quy mô lớn hơn.
Áp dụng cơ chế cập nhật trọng số online.
Giai đoạn 4 (Evaluation):
Sử dụng Log-Loss và AUC để đánh giá độ chính xác của dự đoán.
6. Điểm nhấn để đạt điểm cao
Phân tích độ phức tạp: Chứng minh thuật toán của bạn có độ phức tạp thời gian $O(1)$ cho mỗi bản ghi và bộ nhớ $O(N)$ cố định (nhờ Hashing Trick).
So sánh: So sánh hiệu quả giữa việc train batch truyền thống (Logistic Regression) và Online Learning (FTRL).
Trực quan hóa: Vẽ đồ thị sự thay đổi của Log-Loss theo thời gian khi stream dữ liệu chảy vào.
