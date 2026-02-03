"""
Sơ đồ mối liên hệ: Bài toán tối ưu (Phần 1) ↔ Trạng thái thuật toán (Phần 2) và vai trò của z, n.

Mô tả:
- Phần 1: Mục tiêu (min Log-Loss + L1 + L2) → tìm w.
- Phần 2: Thực thi online → lưu (z, n) thay vì w.
- n_i: Tốc độ học per-coordinate (tích lũy g²).
- z_i: Điều khiển độ thưa (|z_i| ≤ λ₁ → w_i = 0).

Chạy: python _networkx/visualize_ftrl_bai_toan_va_trang_thai.py
Ảnh: _networkx/ftrl_bai_toan_trang_thai.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # --- Phần 1: Bài toán tối ưu ---
    box1 = FancyBboxPatch((0.5, 6.5), 5, 2.8, boxstyle="round,pad=0.05",
                          facecolor='#EBF5FB', edgecolor='#3498DB', linewidth=2)
    ax.add_patch(box1)
    ax.text(3, 8.8, '1. Bài toán tối ưu', fontsize=14, fontweight='bold', ha='center')
    ax.text(3, 8.2, r'Mục tiêu: min$_w$ $\sum_t \ell_t(w) + \lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$', fontsize=10, ha='center')
    ax.text(3, 7.5, 'Tìm bộ trọng số w (Log-Loss + L1 + L2)', fontsize=9, ha='center', style='italic')
    ax.text(3, 6.9, 'Dữ liệu 1TB → không giải trực tiếp w cùng lúc', fontsize=8, ha='center', color='gray')

    # --- Phần 2: Trạng thái thuật toán ---
    box2 = FancyBboxPatch((8, 6.5), 5.5, 2.8, boxstyle="round,pad=0.05",
                          facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2)
    ax.add_patch(box2)
    ax.text(10.75, 8.8, '2. Trạng thái thuật toán', fontsize=14, fontweight='bold', ha='center')
    ax.text(10.75, 8.2, 'Lưu (z, n) thay vì lưu w', fontsize=10, ha='center')
    ax.text(10.75, 7.5, r'$z_i$: tổng gradient đã điều chỉnh  |  $n_i$: $\sum g_{t,i}^2$', fontsize=9, ha='center')
    ax.text(10.75, 6.9, r'w tính lazy khi cần từ z, n và $\lambda_1$, $\lambda_2$, $\alpha$, $\beta$', fontsize=8, ha='center', color='gray')

    # Mũi tên nối Phần 1 → Phần 2
    ax.annotate('', xy=(8, 7.9), xytext=(5.5, 7.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
    ax.text(6.75, 8.15, 'Thực thi online:\nkhông cập nhật w trực tiếp', fontsize=8, ha='center', color='#2C3E50')

    # --- Nhánh n_i: Tốc độ học ---
    box_n = FancyBboxPatch((0.5, 2.8), 4.5, 2.2, boxstyle="round,pad=0.05",
                           facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=1.5)
    ax.add_patch(box_n)
    ax.text(2.75, 4.7, r'Vai trò $n_i$', fontsize=11, fontweight='bold', ha='center', color='#1ABC9C')
    ax.text(2.75, 4.2, r'Tích lũy $n_i \leftarrow n_i + g_{t,i}^2$', fontsize=9, ha='center')
    ax.text(2.75, 3.6, 'Per-coordinate learning rate:', fontsize=9, ha='center')
    ax.text(2.75, 3.1, r'Feature xuất hiện nhiều ($n_i$ lớn) → tốc độ học giảm', fontsize=8, ha='center', wrap=True)
    ax.text(2.75, 2.9, 'Feature hiếm → tốc độ học đủ lớn để nhận diện', fontsize=8, ha='center')

    # Mũi tên từ Trạng thái xuống n_i
    ax.annotate('', xy=(2.75, 5.0), xytext=(10.75, 6.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#1ABC9C', connectionstyle='arc3,rad=-0.2'))
    ax.text(7.2, 5.9, r'$n_i$', fontsize=10, color='#1ABC9C', fontweight='bold')

    # --- Nhánh z_i: Độ thưa ---
    box_z = FancyBboxPatch((8.5, 2.8), 5, 2.2, boxstyle="round,pad=0.05",
                           facecolor='#FDEDEC', edgecolor='#E74C3C', linewidth=1.5)
    ax.add_patch(box_z)
    ax.text(11, 4.7, r'Vai trò $z_i$', fontsize=11, fontweight='bold', ha='center', color='#E74C3C')
    ax.text(11, 4.2, r'Ngưỡng quyết định: so sánh $|z_i|$ với $\lambda_1$', fontsize=9, ha='center')
    ax.text(11, 3.6, r'Nếu $|z_i| \leq \lambda_1$ thì $w_i = 0$ (sparsity)', fontsize=9, ha='center')
    ax.text(11, 3.1, 'Mô hình thưa, tiết kiệm bộ nhớ', fontsize=8, ha='center')
    ax.text(11, 2.9, 'Tổng gradient đã điều chỉnh (adjusted)', fontsize=8, ha='center')

    # Mũi tên từ Trạng thái xuống z_i
    ax.annotate('', xy=(11, 5.0), xytext=(10.75, 6.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#E74C3C', connectionstyle='arc3,rad=0.2'))
    ax.text(10.5, 5.9, r'$z_i$', fontsize=10, color='#E74C3C', fontweight='bold')

    # --- w (lazy) ---
    box_w = FancyBboxPatch((5, 0.5), 4, 1.6, boxstyle="round,pad=0.05",
                           facecolor='#E8DAEF', edgecolor='#9B59B6', linewidth=1.5)
    ax.add_patch(box_w)
    ax.text(7, 1.75, r'Trọng số $w_i$ (tính lazy)', fontsize=11, fontweight='bold', ha='center', color='#9B59B6')
    ax.text(7, 1.2, r'Từ $z_i, n_i$ và $\alpha, \beta, \lambda_1, \lambda_2$', fontsize=9, ha='center')
    ax.text(7, 0.7, r'$|z_i| \leq \lambda_1$ thì $w_i=0$; ngược lại dùng công thức FTRL', fontsize=8, ha='center')

    # Mũi tên từ n_i và z_i xuống w
    ax.annotate('', xy=(6.2, 1.1), xytext=(2.75, 2.8),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#9B59B6', connectionstyle='arc3,rad=0.15'))
    ax.annotate('', xy=(7.8, 1.1), xytext=(11, 2.8),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#9B59B6', connectionstyle='arc3,rad=-0.15'))
    ax.text(5, 2.0, 'Công thức', fontsize=8, color='#9B59B6')
    ax.text(9, 2.0, 'Công thức', fontsize=8, color='#9B59B6')

    ax.set_title('Mối liên hệ: Bài toán tối ưu (1) ↔ Trạng thái thuật toán (2)\nVai trò của $z_i$ (độ thưa) và $n_i$ (tốc độ học)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")


if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ftrl_bai_toan_trang_thai.png')
    draw_diagram(out_path)
    print("Mở _networkx/ftrl_bai_toan_trang_thai.png để xem.")
