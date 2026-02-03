"""
Sơ đồ NetworkX/matplotlib: Vòng lặp xử lý một mẫu (Online Learning Loop) FTRL-Proximal.

Mô tả:
- 5 bước: Dự đoán → Gradient → Cập nhật n_i → Tính sigma_i → Cập nhật z_i
- Công thức tính w_i (closed-form) và tham số
- Luồng dữ liệu giữa các bước

Chạy: python _networkx/visualize_ftrl_online_loop.py
Ảnh: _networkx/ftrl_online_loop.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch

def draw_flowchart(output_path: str):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # ---- INPUT ----
    box_in = FancyBboxPatch((0.5, 10.5), 3.5, 1.4, boxstyle="round,pad=0.04",
                             facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=1.5)
    ax.add_patch(box_in)
    ax.text(2.25, 11.5, 'Input', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.25, 10.95, r'$(x_t, y_t)$  +  $(z_i, n_i)$', fontsize=10, ha='center')

    # ---- CÔNG THỨC w_i (closed-form) ----
    box_w = FancyBboxPatch((12, 9.2), 4.5, 3.2, boxstyle="round,pad=0.04",
                            facecolor='#EBF5FB', edgecolor='#3498DB', linewidth=1.5)
    ax.add_patch(box_w)
    ax.text(14.25, 12.15, r'Công thức $w_i$ (closed-form)', fontsize=11, fontweight='bold', ha='center')
    ax.text(14.25, 11.6, r'$|z_i| \leq \lambda_1 \Rightarrow w_i = 0$', fontsize=9, ha='center')
    ax.text(14.25, 11.0, r'Ngược lại: $w_i = -\frac{z_i - \mathrm{sgn}(z_i)\lambda_1}{(\beta+\sqrt{n_i})/\alpha + \lambda_2}$', fontsize=8, ha='center')
    ax.text(14.25, 10.35, r'L1: ngưỡng sparsity  |  L2: tránh overfit', fontsize=8, ha='center', color='gray')
    ax.text(14.25, 9.7, r'$(\beta+\sqrt{n_i})/\alpha$: per-coord. LR', fontsize=8, ha='center', color='gray')

    # ---- BƯỚC 1: Dự đoán ----
    box1 = FancyBboxPatch((0.5, 7.8), 5.2, 2.0, boxstyle="round,pad=0.04",
                           facecolor='#FDEBD0', edgecolor='#E67E22', linewidth=1.5)
    ax.add_patch(box1)
    ax.text(3.1, 9.5, 'Bước 1: Dự đoán', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.1, 9.0, r'$w_{t,i}$ từ $(z_i, n_i)$ (công thức bên phải)', fontsize=9, ha='center')
    ax.text(3.1, 8.4, r'score $= \sum_i w_{t,i} x_{t,i}$  $\Rightarrow$  $p_t = \sigma(\mathrm{score})$', fontsize=9, ha='center')
    ax.text(3.1, 8.0, 'Output: $p_t$, $w_{t,i}$', fontsize=8, ha='center', style='italic')

    # ---- BƯỚC 2: Gradient ----
    box2 = FancyBboxPatch((0.5, 5.5), 5.2, 1.9, boxstyle="round,pad=0.04",
                           facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=1.5)
    ax.add_patch(box2)
    ax.text(3.1, 7.15, 'Bước 2: Tính gradient', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.1, 6.6, r'$g_t = p_t - y_t$  ;  $g_{t,i} = g_t \cdot x_{t,i}$', fontsize=10, ha='center')
    ax.text(3.1, 5.95, 'Output: $g_t$, $g_{t,i}$', fontsize=8, ha='center', style='italic')

    # ---- BƯỚC 3: Cập nhật n_i ----
    box3 = FancyBboxPatch((0.5, 3.8), 5.2, 1.4, boxstyle="round,pad=0.04",
                           facecolor='#D1F2EB', edgecolor='#1ABC9C', linewidth=1.5)
    ax.add_patch(box3)
    ax.text(3.1, 5.0, 'Bước 3: Cập nhật $n_i$', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.1, 4.45, r'$n_i \leftarrow n_i + g_{t,i}^2$  (per-coordinate learning rate)', fontsize=9, ha='center')
    ax.text(3.1, 3.95, 'Output: $n_i$ mới', fontsize=8, ha='center', style='italic')

    # ---- BƯỚC 4: Tính sigma_i ----
    box4 = FancyBboxPatch((0.5, 2.2), 5.2, 1.3, boxstyle="round,pad=0.04",
                           facecolor='#E8DAEF', edgecolor='#9B59B6', linewidth=1.5)
    ax.add_patch(box4)
    ax.text(3.1, 3.35, r'Bước 4: Tính $\sigma_i$', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.1, 2.8, r'$\sigma_i = (\sqrt{n_i^{\mathrm{new}}} - \sqrt{n_i^{\mathrm{old}}}) / \alpha$', fontsize=9, ha='center')
    ax.text(3.1, 2.35, 'Output: $\\sigma_i$', fontsize=8, ha='center', style='italic')

    # ---- BƯỚC 5: Cập nhật z_i ----
    box5 = FancyBboxPatch((0.5, 0.4), 5.2, 1.5, boxstyle="round,pad=0.04",
                           facecolor='#FADBD8', edgecolor='#C0392B', linewidth=1.5)
    ax.add_patch(box5)
    ax.text(3.1, 1.7, r'Bước 5: Cập nhật $z_i$', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.1, 1.15, r'$z_i \leftarrow z_i + g_{t,i} - \sigma_i w_{t,i}$', fontsize=10, ha='center')
    ax.text(3.1, 0.65, 'Output: $z_i$ mới  (trạng thái cho mẫu tiếp theo)', fontsize=8, ha='center', style='italic')

    # ---- Tham số (bảng) ----
    box_param = FancyBboxPatch((12, 5.2), 4.5, 3.6, boxstyle="round,pad=0.04",
                               facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=1.5)
    ax.add_patch(box_param)
    ax.text(14.25, 8.6, 'Tham số FTRL-Proximal', fontsize=11, fontweight='bold', ha='center')
    params = [(r'$\alpha$', 'Learning rate cơ bản', '0.1'),
              (r'$\beta$', 'Làm mượt LR giai đoạn đầu', '1.0'),
              (r'$\lambda_1$ (L1)', 'Ngưỡng sparsity', '1.0'),
              (r'$\lambda_2$ (L2)', 'Cường độ L2', '1.0')]
    y0 = 8.15
    for i, (sym, meaning, default) in enumerate(params):
        ax.text(12.4, y0 - i * 0.55, sym, fontsize=10, fontweight='bold')
        ax.text(13.8, y0 - i * 0.55, meaning, fontsize=8)
        ax.text(16.2, y0 - i * 0.55, default, fontsize=8, ha='right')

    # ---- Ghi chú sparsity ----
    ax.text(3.1, -0.35, 'Chỉ các feature có trong vector thưa $x_t$ mới được cập nhật  ;  Chi phí: O(số phần tử khác 0)', fontsize=9, ha='center', color='gray')

    # ========== MŨI TÊN ==========
    def arrow(ax, start, end, label=None, color='#2C3E50'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color=color))
        if label:
            mid = ((start[0]+end[0])/2, (start[1]+end[1])/2)
            ax.text(mid[0], mid[1], label, fontsize=8, ha='center', color=color)

    # Input -> Bước 1
    arrow(ax, (2.25, 10.5), (3.1, 9.8), r'$x_t, z_i, n_i$')
    # Công thức w -> Bước 1 (nét đứt hoặc từ bên phải)
    ax.annotate('', xy=(5.7, 8.8), xytext=(12, 10.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#3498DB', linestyle='--'))
    ax.text(8.5, 9.7, r'$w_{t,i}$', fontsize=9, color='#3498DB', fontweight='bold')
    # Bước 1 -> Bước 2
    arrow(ax, (3.1, 7.8), (3.1, 7.4), r'$p_t$')
    # Bước 2 -> Bước 3
    arrow(ax, (3.1, 5.5), (3.1, 5.2), r'$g_{t,i}$')
    # Bước 3 -> Bước 4
    arrow(ax, (3.1, 3.8), (3.1, 3.5), r'$n_i$ mới')
    # Bước 4 -> Bước 5
    arrow(ax, (3.1, 2.2), (3.1, 1.9), r'$\sigma_i$')
    # Bước 5 -> (vòng lại: z_i, n_i cho mẫu tiếp)
    ax.annotate('', xy=(2.25, 10.5), xytext=(3.1, 0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#27AE60', connectionstyle='arc3,rad=0.3'))
    ax.text(1.2, 5.2, r'$(z_i, n_i)$ mới', fontsize=8, color='#27AE60', rotation=90)

    ax.set_title('3. Vòng lặp xử lý một mẫu (Online Learning Loop) — FTRL-Proximal\nLuồng: Dự đoán $\\rightarrow$ Gradient $\\rightarrow$ Cập nhật $n_i$ $\\rightarrow$ $\\sigma_i$ $\\rightarrow$ Cập nhật $z_i$', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")


if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ftrl_online_loop.png')
    draw_flowchart(out_path)
    print("Mở _networkx/ftrl_online_loop.png để xem.")
