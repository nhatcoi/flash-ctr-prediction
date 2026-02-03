# Minh họa vector thưa và ma trận thưa

Thư mục này chứa script trực quan hóa **vector thưa** (output Feature Hashing) và **ma trận thưa** (nhiều vector thưa).

## Yêu cầu

- Python 3.8+
- `matplotlib`, `numpy`

## Script chính (đẹp, trực quan)

```bash
python _networkx/visualize_sparse_vector_and_matrix.py
```

**Kết quả:**
- **sparse_vector_viz.png**: Một vector thưa — đồ thị dạng sao: nút trung tâm "Vector thưa", các nút bucket xung quanh với nhãn (bucket_id, value). Màu xanh = value dương, đỏ = âm.
- **sparse_matrix_viz.png**: Ma trận thưa — đồ thị hai phần (bipartite): bên trái = các mẫu (dòng), bên phải = các bucket (cột); cạnh nối = phần tử khác 0, độ đậm/màu theo giá trị.

## Tổng quan kiến trúc theo tầng (4 tầng)

```bash
python _networkx/visualize_architecture_layers.py
```

**architecture_layers.png**: Sơ đồ luồng 4 tầng — (1) **Tầng dữ liệu:** File .gz/.txt → CriteoDataLoader → StreamingIterator. (2) **Tầng tiền xử lý:** Xử lý thiếu → Log transform → Feature Hashing → Vector thưa {bucket: value}. (3) **Tầng thuật toán:** FTRL-Proximal (chính), Online Logistic Regression (so sánh). (4) **Tầng huấn luyện & đánh giá:** StreamingTrainer → RunningMetrics & Visualizer, Mô hình .pkl. Dùng NetworkX (DiGraph) khi có sẵn; fallback matplotlib.

---

## Sơ đồ FTRL: Bài toán tối ưu ↔ Trạng thái (z, n)

```bash
python _networkx/visualize_ftrl_bai_toan_va_trang_thai.py
```

**ftrl_bai_toan_trang_thai.png**: Mối liên hệ giữa Phần 1 (Bài toán tối ưu: min Log-Loss + L1 + L2) và Phần 2 (Trạng thái: lưu z, n thay vì w); vai trò của **n_i** (tốc độ học per-coordinate) và **z_i** (ngưỡng sparsity: |z_i| ≤ λ₁ → w_i = 0); w tính lazy từ z, n.

---

## Vòng lặp xử lý một mẫu (5 bước FTRL)

```bash
python _networkx/visualize_ftrl_online_loop.py
```

**ftrl_online_loop.png**: Sơ đồ luồng 5 bước — (1) Dự đoán (w_t,i, score, p_t), (2) Gradient (g_t, g_t,i), (3) Cập nhật n_i, (4) Tính σ_i, (5) Cập nhật z_i; kèm công thức closed-form tính w_i và bảng tham số α, β, λ₁, λ₂. Mũi tên thể hiện luồng dữ liệu và vòng lặp (z, n mới → input mẫu tiếp theo).

---

## Minh họa thực tế: Mô hình .pkl có thể làm gì?

```bash
python _networkx/visualize_pkl_usage.py
```

**ftrl_pkl_usage.png**: Hai luồng sử dụng file .pkl — (1) **Inference:** Load .pkl → Request (user, ad, context) → Preprocessor → vector thưa → model.predict() → p (CTR) → Dùng p (ranking, bidding, lọc, A/B test). (2) **Incremental learning:** Load .pkl → Luồng dữ liệu mới → model.update() → Save .pkl. Có so sánh với LLM và gợi ý lệnh evaluate.

---

## Script cũ (có fallback khi thiếu NetworkX)

```bash
python _networkx/visualize_sparse_vector.py
```

- **sparse_vector.png**: Vector thưa mẫu.
- **sparse_vector_real.png**: Vector thưa từ 1 mẫu thật (nếu có `data/sample/train.txt`).
