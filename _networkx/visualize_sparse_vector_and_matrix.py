"""
Mô phỏng trực quan: (1) Vector thưa, (2) Ma trận thưa (nhiều vector thưa).
Dùng matplotlib để vẽ đẹp, không phụ thuộc NetworkX (tương thích Python 3.14).

Chạy: python _networkx/visualize_sparse_vector_and_matrix.py
Ảnh: _networkx/sparse_vector_viz.png, _networkx/sparse_matrix_viz.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm

# --- Dữ liệu mẫu ---

def get_sample_sparse_vector():
    """Một vector thưa (1 mẫu)."""
    return {
        218608: 1.0,
        153521: 0.693,
        175177: 1.792,
        50667: 7.232,
        44102: 1.386,
        89201: -1.0,
        200441: 1.0,
        102333: 1.0,
        145672: -1.0,
        88123: 1.0,
    }


def get_sample_sparse_matrix(num_samples=5, num_buckets=12):
    """
    Ma trận thưa: list of dict (mỗi dict = 1 vector thưa).
    Trả về: list of dict, và danh sách bucket xuất hiện ít nhất 1 lần.
    """
    buckets_pool = [218608, 153521, 175177, 50667, 44102, 89201, 200441, 102333, 145672, 88123, 111111, 222222]
    matrix = []
    for i in range(num_samples):
        n_active = np.random.randint(4, 9)
        chosen = np.random.choice(len(buckets_pool), size=n_active, replace=False)
        vec = {}
        for j in chosen:
            bid = buckets_pool[j]
            vec[bid] = round(np.random.uniform(-1.5, 2.0), 2)
            if vec[bid] == 0:
                vec[bid] = 0.5
        matrix.append(vec)
    all_buckets = sorted(set(b for vec in matrix for b in vec.keys()))
    return matrix, all_buckets


# --- Vẽ 1 vector thưa (dạng sao, đẹp) ---

def draw_sparse_vector(sparse_vec: dict, output_path: str):
    sparse_vec = {k: v for k, v in sparse_vec.items() if v != 0}
    if not sparse_vec:
        return
    buckets = list(sparse_vec.keys())
    values = list(sparse_vec.values())
    n = len(buckets)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # Layout: nút trung tâm "Vector thưa", các bucket trên vòng tròn
    center = (0, 0)
    r = 2.2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {0: center}
    for i, a in enumerate(angles):
        positions[i + 1] = (r * np.cos(a), r * np.sin(a))

    # Màu theo value: dương = xanh lá, âm = cam
    vmin, vmax = min(values), max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors_pos = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
    sizes = [300 + 180 * min(abs(v), 2) for v in values]

    # Vẽ cạnh (center -> bucket)
    for i in range(n):
        ax.plot([center[0], positions[i + 1][0]], [center[1], positions[i + 1][1]],
                color='#95a5a6', lw=2.5, alpha=0.7, zorder=0)
    # Nút trung tâm
    ax.scatter([center[0]], [center[1]], s=2200, c='#f1c40f', edgecolors='#2c3e50', linewidths=2, zorder=2)
    ax.text(center[0], center[1], 'Vector thưa\n(1 mẫu)', ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    # Nút bucket
    xs = [positions[i + 1][0] for i in range(n)]
    ys = [positions[i + 1][1] for i in range(n)]
    ax.scatter(xs, ys, s=sizes, c=colors_pos, edgecolors='#2c3e50', linewidths=1.2, alpha=0.9, zorder=2)
    for i in range(n):
        ax.annotate(f'bucket {buckets[i]}\n{values[i]:.2f}', xy=positions[i + 1], fontsize=8,
                    ha='center', va='center', fontweight='bold')
    ax.set_title('Vector thưa (1 mẫu)\nDict[bucket_id → value], chỉ lưu phần tử ≠ 0', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")


# --- Vẽ ma trận thưa (bipartite: mẫu bên trái, bucket bên phải) ---

def draw_sparse_matrix(matrix: list, all_buckets: list, output_path: str):
    """
    matrix: list of dict (mỗi dict = 1 vector thưa).
    all_buckets: danh sách bucket (cột).
    """
    n_samples = len(matrix)
    n_buckets = len(all_buckets)
    bucket_to_col = {b: j for j, b in enumerate(all_buckets)}

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    # Layout: mẫu bên trái (cột 0), bucket bên phải (cột 1)
    left_x, right_x = -2.5, 2.5
    # Mẫu: phân bố đều theo chiều dọc
    sample_ys = np.linspace(1.5, -1.5, n_samples) if n_samples > 1 else [0]
    # Bucket: phân bố đều
    bucket_ys = np.linspace(1.5, -1.5, n_buckets) if n_buckets > 1 else [0]

    # Vẽ cạnh (mẫu i — bucket j) nếu matrix[i][bucket_j] != 0
    lines = []
    line_values = []
    for i, vec in enumerate(matrix):
        y_s = sample_ys[i] if n_samples > 1 else sample_ys[0]
        for bid, val in vec.items():
            if val == 0:
                continue
            if bid not in bucket_to_col:
                continue
            j = bucket_to_col[bid]
            y_b = bucket_ys[j] if n_buckets > 1 else bucket_ys[0]
            lines.append([[left_x, y_s], [right_x, y_b]])
            line_values.append(val)

    if lines:
        v_abs = max(abs(v) for v in line_values) or 1
        # Màu: âm = đỏ, dương = xanh (normalize về [0,1] cho RdYlGn: 0=đỏ, 0.5=vàng, 1=xanh)
        line_colors = [plt.cm.RdYlGn(0.5 + 0.5 * (v / v_abs)) for v in line_values]
        lc = LineCollection(lines, colors=line_colors,
                            linewidths=[2 + 0.8 * min(abs(v), 2) for v in line_values], alpha=0.75)
        ax.add_collection(lc)

    # Nút mẫu (trái)
    ax.scatter([left_x] * len(sample_ys), sample_ys, s=600, c='#3498db', edgecolors='#2c3e50', linewidths=1.5, zorder=2)
    for i, y in enumerate(sample_ys):
        ax.text(left_x - 0.35, y, f'Mẫu {i+1}', ha='right', va='center', fontsize=9, fontweight='bold')
    # Nút bucket (phải)
    ax.scatter([right_x] * len(bucket_ys), bucket_ys, s=400, c='#9b59b6', edgecolors='#2c3e50', linewidths=1.2, zorder=2)
    for j, (y, bid) in enumerate(zip(bucket_ys, all_buckets)):
        ax.text(right_x + 0.35, y, str(bid), ha='left', va='center', fontsize=8)
    # Nhãn hai nhóm
    ax.text(left_x, 2.0, 'Mẫu (dòng)', ha='center', fontsize=11, fontweight='bold')
    ax.text(right_x, 2.0, 'Bucket (cột)', ha='center', fontsize=11, fontweight='bold')
    ax.set_xlim(left_x - 1.2, right_x + 1.2)
    ax.set_ylim(-2.2, 2.4)
    ax.set_title('Ma trận thưa (nhiều vector thưa)\nCạnh = phần tử khác 0; độ đậm/màu theo giá trị', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")


if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Vector thưa
    vec = get_sample_sparse_vector()
    draw_sparse_vector(vec, os.path.join(out_dir, 'sparse_vector_viz.png'))

    # 2. Ma trận thưa
    np.random.seed(42)
    matrix, all_buckets = get_sample_sparse_matrix(num_samples=5, num_buckets=10)
    draw_sparse_matrix(matrix, all_buckets, os.path.join(out_dir, 'sparse_matrix_viz.png'))

    print('Xong. Mở _networkx/sparse_vector_viz.png và sparse_matrix_viz.png để xem.')
