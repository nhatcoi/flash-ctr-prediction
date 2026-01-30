# Project Commands Summary

## 1. Môi trường (Environment)
```bash
# Tạo môi trường ảo
python3 -m venv venv

# Kích hoạt (macOS/Linux)
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
pip install huggingface_hub
```

## 2. Dữ liệu từ Hugging Face
```bash
# Đăng nhập (Lấy token tại hf.co/settings/tokens)
hf auth login

# Tải file dữ liệu (Ví dụ: ngày 2)
hf download criteo/CriteoClickLogs day_2.gz --repo-type dataset --local-dir ./data

# Xem nhanh dữ liệu nén (macOS)
gzcat data/day_2.gz | head -n 10
```

## 3. Chạy dự án với dữ liệu thật (Real Data)
> **Lưu ý:** Dữ liệu thật rất lớn (~40 triệu dòng/file), nên dùng `--max-samples` để giới hạn khi mới bắt đầu.

```bash
# Huấn luyện trên 1 triệu dòng từ file nén ngày 2
python main.py --train --data data/day_2.gz --max-samples 1000000 --output models/ftrl_big.pkl

# Đánh giá mô hình đã huấn luyện trên file ngày 3
python main.py --evaluate --data data/day_3.gz --model models/ftrl_big.pkl --max-samples 100000

# So sánh 2 thuật toán trên dữ liệu thật (vẽ đồ thị)
python main.py --compare --data data/day_2.gz --test-data data/day_3.gz --max-samples 200000 --plot
```

## 4. Chạy dự án mẫu (Demo & Test)
```bash
# Chạy Demo tổng hợp với dữ liệu sample tự sinh
python main.py --demo

# Phân tích đồ thị đặc trưng (Numerical features correlation)
python main.py --graph --data data/day_2.gz --max-samples 10000 --plot
```

## 5. Tham số chính (Main Arguments)
- `--train`: Chế độ huấn luyện.
- `--evaluate`: Chế độ đánh giá.
- `--compare`: So sánh FTRL vs Online Logistic Regression.
- `--data [path]`: Đường dẫn tới file (hỗ trợ cả .txt và .gz).
- `--max-samples [n]`: Giới hạn số dòng xử lý (Rất quan trọng với data 1TB).
- `--plot`: Tự động vẽ và lưu đồ thị vào thư mục `outputs/`.
