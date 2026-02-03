"""
Sơ đồ minh họa thực tế: Mô hình .pkl có thể làm gì?

Hai luồng chính:
1. Inference (dự đoán): Load .pkl → Request → Preprocessor → predict → p (CTR) → Ranking/Bidding/Filter
2. Incremental learning: Load .pkl → Luồng dữ liệu mới → update → Save .pkl

Chạy: python _networkx/visualize_pkl_usage.py
Ảnh: _networkx/ftrl_pkl_usage.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')

    # ---- File .pkl (trung tâm) ----
    box_pkl = FancyBboxPatch((6.5, 8.2), 5, 2.4, boxstyle="round,pad=0.05",
                             facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2)
    ax.add_patch(box_pkl)
    ax.text(9, 10.35, 'File mô hình (.pkl)', fontsize=13, fontweight='bold', ha='center')
    ax.text(9, 9.75, 'Nội dung: z, n, alpha, beta, L1, L2, num_updates', fontsize=9, ha='center')
    ax.text(9, 9.15, r'$w$ tính lazy từ $(z, n)$ khi predict', fontsize=9, ha='center')
    ax.text(9, 8.55, 'Input: vector thưa  →  Output: $p \\in [0,1]$ (CTR)', fontsize=8, ha='center', color='gray')

    # ---- Nhánh 1: INFERENCE (Dự đoán) ----
    ax.text(2, 10.8, '1. Dự đoán (Inference)', fontsize=12, fontweight='bold', color='#27AE60')
    steps1 = [
        (1.5, 9.8, 'Load model\nFTRLProximal.load(.pkl)', '#D5F5E3'),
        (1.5, 8.6, 'Request\n(user X, ad Y, 10h, mobile)', '#D5F5E3'),
        (1.5, 7.4, 'Preprocessor\n(cùng config hash)\n→ vector thưa', '#A9DFBF'),
        (1.5, 5.8, 'model.predict(features)\n→ $p$ (xác suất click)', '#58D68D'),
        (1.5, 4.2, 'Dùng $p$ trong nghiệp vụ', '#27AE60'),
    ]
    for x, y, label, color in steps1:
        b = FancyBboxPatch((x - 0.9, y - 0.35), 1.8, 0.7, boxstyle="round,pad=0.03",
                           facecolor=color, edgecolor='#27AE60', linewidth=1)
        ax.add_patch(b)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', wrap=True)
    # Mũi tên nối các bước inference
    for i in range(len(steps1) - 1):
        ax.annotate('', xy=(1.5, steps1[i+1][1] + 0.35), xytext=(1.5, steps1[i][1] - 0.35),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#27AE60'))
    # Mũi tên từ .pkl sang "Load model"
    ax.annotate('', xy=(6.5, 9.4), xytext=(3.3, 9.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))
    ax.text(4.7, 9.7, 'load', fontsize=8, color='#2C3E50')
    # Mũi tên từ "vector thưa" vào .pkl (predict)
    ax.annotate('', xy=(6.5, 8.8), xytext=(3.3, 7.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#27AE60', connectionstyle='arc3,rad=0.15'))
    ax.text(4.6, 8.0, 'features', fontsize=8, color='#27AE60')
    # p ra khỏi .pkl (kết quả predict)
    ax.annotate('', xy=(3.3, 5.8), xytext=(6.5, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#27AE60', connectionstyle='arc3,rad=-0.2'))
    ax.text(5.2, 7.4, '$p$', fontsize=9, color='#27AE60', fontweight='bold')
    # Box chi tiết "Dùng p"
    box_use = FancyBboxPatch((0.3, 3.2), 2.4, 0.75, boxstyle="round,pad=0.03",
                             facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=1)
    ax.add_patch(box_use)
    ax.text(1.5, 3.6, 'Ranking ad (p cao → ưu tiên)', fontsize=7, ha='center')
    ax.text(1.5, 3.25, 'Bidding, Lọc p thấp, A/B test', fontsize=7, ha='center')

    # ---- Nhánh 2: HỌC TIẾP (Incremental) ----
    ax.text(14.5, 10.8, '2. Học tiếp (Incremental)', fontsize=12, fontweight='bold', color='#3498DB')
    steps2 = [
        (16.5, 9.8, 'Load .pkl', '#D6EAF8'),
        (16.5, 8.8, 'Luồng dữ liệu mới\n(click / no-click)', '#AED6F1'),
        (16.5, 7.5, 'model.update(features, label)\ncho từng mẫu', '#3498DB'),
        (16.5, 6.2, 'Định kỳ save .pkl', '#2E86AB'),
    ]
    for x, y, label, color in steps2:
        b = FancyBboxPatch((x - 0.95, y - 0.35), 1.9, 0.7, boxstyle="round,pad=0.03",
                           facecolor=color, edgecolor='#3498DB', linewidth=1)
        ax.add_patch(b)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', wrap=True)
    for i in range(len(steps2) - 1):
        ax.annotate('', xy=(16.5, steps2[i+1][1] + 0.35), xytext=(16.5, steps2[i][1] - 0.35),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#3498DB'))
    # .pkl -> Load (nhánh 2)
    ax.annotate('', xy=(11.5, 9.4), xytext=(15.6, 9.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2C3E50'))
    ax.text(13.4, 9.7, 'load', fontsize=8, color='#2C3E50')
    # Save -> .pkl
    ax.annotate('', xy=(11.5, 8.5), xytext=(15.6, 6.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#3498DB', connectionstyle='arc3,rad=-0.25'))
    ax.text(13.2, 7.2, 'save', fontsize=8, color='#3498DB')

    # ---- So sánh LLM (góc dưới) ----
    box_llm = FancyBboxPatch((6, 0.2), 6, 1.8, boxstyle="round,pad=0.04",
                             facecolor='#F5EEF8', edgecolor='#9B59B6', linewidth=1)
    ax.add_patch(box_llm)
    ax.text(9, 1.7, 'So sánh: LLM nhận text → sinh text  |  FTRL (.pkl) nhận vector thưa → trả về $p$ (CTR)', fontsize=9, ha='center')
    ax.text(9, 1.0, 'Đánh giá: python main.py --evaluate --model models/ftrl.pkl --data test.txt  →  predict (không update) → Log-Loss, AUC', fontsize=8, ha='center', color='gray')
    ax.text(9, 0.4, 'Mô hình .pkl = dự đoán xác suất click; dùng để rank, bid, lọc hoặc học tiếp.', fontsize=8, ha='center', style='italic')

    ax.set_title('Mô hình .pkl có thể làm gì? — Inference (dự đoán CTR) và Incremental Learning', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")


if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ftrl_pkl_usage.png')
    draw_diagram(out_path)
    print("Mở _networkx/ftrl_pkl_usage.png để xem.")
