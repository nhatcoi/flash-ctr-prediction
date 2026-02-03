# Đánh giá, Chỉ số và Chiến lược Kiểm thử (Evaluation & Metrics)

Chương này mô tả cách hệ thống đo lường độ chính xác và tại sao cấu trúc dữ liệu Criteo lại yêu cầu một chiến lược kiểm thử đặc thù.

## 1. Công cụ đo lường (Metrics Implementation)

### Lớp thực thi: `RunningMetrics`
- **Vị trí**: `src/evaluation/metrics.py`
- **Cơ chế**: Tính toán "online" (không lưu toàn bộ kết quả vào mảng để tránh tràn RAM).

#### A. Logarithmic Loss (Hàm `log_loss`)
- **Công thức**: $L = -(y \log(p) + (1-y) \log(1-p))$
- **Đặc điểm**: Phạt nặng các dự đoán sai mà mô hình đang "tự tin" (ví dụ dự đoán 0.9 nhưng nhãn thật là 0). 

#### B. AUC - Area Under ROC Curve (Hàm `auc_score`)
- **Cơ chế**: Sắp xếp các mẫu theo xác suất dự đoán và tính toán khả năng phân loại.
- **Ý nghĩa**: AUC = 0.74 cho thấy mô hình có khả năng xếp hạng quảng cáo tiềm năng tốt hơn nhiều so với việc chọn ngẫu nhiên.

#### C. Độ thưa (Sparsity)
- **Hàm**: `model.sparsity()`
- **Logic**: `1.0 - (count_nonzero(w) / hash_space)`.
- **Mục tiêu**: Sparsity $> 85\%$ là minh chứng cho việc thuật toán FTRL đã loại bỏ nhiễu thành công.

---

## 2. Chiến lược kiểm thử Day-by-Day

Tại sao dự án dùng **Day 2** để train và **Day 3** để đánh giá?

### A. Mô phỏng Thực tế (Real-world Simulation)
Trong hệ thống quảng cáo, chúng ta dùng dữ liệu quá khứ (hôm qua) để phục vụ dự đoán cho tương lai (hôm nay). Việc đánh giá trên Day 3 đảm bảo mô hình có khả năng **Tổng quát hóa (Generalization)** trên dữ liệu mới hoàn toàn.

### B. Chế độ "Đi thi" (Inference Mode)
Khi chạy lệnh `--evaluate`, mô hình sẽ ở trạng thái **Read-only**:
1. Load file `.pkl`.
2. Duyệt qua từng dòng của Day 3.
3. Chỉ chạy hàm `predict` (không chạy hàm `update`).
4. Kết quả dự đoán được đối chiếu với nhãn thật để ra con số cuối cùng.

---

## 3. Cấu trúc đầu ra Kết quả (Outputs)

Mục đích của việc xử lý kết quả đầu ra:
1.  **Biểu đồ đường (Line Plot)**: Trích xuất lịch sử từ `history` của trainer để vẽ đường hôi tụ của Loss.
2.  **So sánh (Comparison Table)**: Đối chiếu FTRL (Sparse) và Online LR (Dense) để thấy sự khác biệt về tài nguyên RAM và file size mô hình.
3.  **NetworkX Graph**: Phân tích sự tương quan giữa các trường thông tin (Feature Interaction).

Dựa vào các chỉ số này, chúng ta có thể khẳng định mô hình đã sẵn sàng cho việc triển khai thực tế trên hệ thống Serving.
