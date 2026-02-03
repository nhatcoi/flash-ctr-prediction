"""
Sơ đồ NetworkX: Tổng quan kiến trúc theo tầng (4 tầng).

Luồng xử lý:
1. Tầng dữ liệu: File .gz/.txt, CriteoDataLoader, StreamingIterator
2. Tầng tiền xử lý: Xử lý thiếu, Log transform, Feature Hashing → vector thưa
3. Tầng thuật toán: FTRL-Proximal, Online Logistic Regression
4. Tầng huấn luyện & đánh giá: StreamingTrainer, metrics, .pkl

Chạy: python _networkx/visualize_architecture_layers.py
Ảnh: _networkx/architecture_layers.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Dữ liệu cho sơ đồ (dùng chung cho cả NetworkX và fallback matplotlib)
LAYER0 = ['File .gz/.txt']
LAYER1 = ['CriteoDataLoader', 'StreamingIterator']
LAYER2 = ['Xử lý thiếu', 'Log transform', 'Feature Hashing', 'Vector thưa\n{bucket: value}']
LAYER3 = ['FTRL-Proximal\n(chính)', 'Online Logistic\nRegression (so sánh)']
LAYER4 = ['StreamingTrainer', 'RunningMetrics\n& Visualizer', 'Mô hình .pkl']

EDGES = [
    ('File .gz/.txt', 'CriteoDataLoader'),
    ('CriteoDataLoader', 'StreamingIterator'),
    ('StreamingIterator', 'Xử lý thiếu'),
    ('Xử lý thiếu', 'Log transform'),
    ('Log transform', 'Feature Hashing'),
    ('Feature Hashing', 'Vector thưa\n{bucket: value}'),
    ('Vector thưa\n{bucket: value}', 'FTRL-Proximal\n(chính)'),
    ('Vector thưa\n{bucket: value}', 'Online Logistic\nRegression (so sánh)'),
    ('FTRL-Proximal\n(chính)', 'StreamingTrainer'),
    ('Online Logistic\nRegression (so sánh)', 'StreamingTrainer'),
    ('StreamingTrainer', 'RunningMetrics\n& Visualizer'),
    ('StreamingTrainer', 'Mô hình .pkl'),
]

COLORS = ['#D5F5E3', '#D6EAF8', '#FDEBD0', '#FADBD8', '#E8DAEF']


def _get_pos():
    """Vị trí node theo tầng (dùng cho cả nx và matplotlib)."""
    layers = [LAYER0, LAYER1, LAYER2, LAYER3, LAYER4]
    y_positions = [4.0, 3.0, 2.0, 1.0, 0.0]
    pos = {}
    for layer, y in zip(layers, y_positions):
        n = len(layer)
        for i, node in enumerate(layer):
            x = 2.0 + (i - (n - 1) / 2) * 1.4
            pos[node] = (x, y)
    return pos


def draw_with_networkx(output_path: str):
    """Vẽ sơ đồ bằng NetworkX (DiGraph + layered pos)."""
    import networkx as nx
    G = nx.DiGraph()
    all_nodes = LAYER0 + LAYER1 + LAYER2 + LAYER3 + LAYER4
    for n in all_nodes:
        G.add_node(n)
    for u, v in EDGES:
        G.add_edge(u, v)
    pos = _get_pos()
    layers = [LAYER0, LAYER1, LAYER2, LAYER3, LAYER4]
    node_colors = []
    for node in G.nodes():
        for idx, layer in enumerate(layers):
            if node in layer:
                node_colors.append(COLORS[idx])
                break
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=2400,
        node_shape='s', alpha=0.95, edgecolors='#2C3E50', linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edge_color='#2C3E50', width=2, arrows=True, arrowsize=22,
        arrowstyle='-|>', connectionstyle='arc3,rad=0.05', ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    _add_tier_labels(ax)
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.6, 4.6)
    ax.set_title(
        'Tổng quan kiến trúc theo tầng\n'
        'Luồng xử lý: Dữ liệu → Tiền xử lý → Thuật toán → Huấn luyện & đánh giá',
        fontsize=13, fontweight='bold',
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu (NetworkX): {output_path}")


def _add_tier_labels(ax):
    ax.text(0.35, 3.5, 'Tầng dữ liệu', fontsize=11, fontweight='bold', va='center', color='#1A5276')
    ax.text(0.35, 2.0, 'Tầng tiền xử lý', fontsize=11, fontweight='bold', va='center', color='#B7950B')
    ax.text(0.35, 1.0, 'Tầng thuật toán', fontsize=11, fontweight='bold', va='center', color='#922B21')
    ax.text(0.35, 0.0, 'Tầng huấn luyện\n& đánh giá', fontsize=11, fontweight='bold', va='center', color='#6C3483')


def draw_with_matplotlib(output_path: str):
    """Fallback: vẽ sơ đồ bằng matplotlib (FancyBboxPatch + mũi tên)."""
    pos = _get_pos()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.6, 4.6)
    ax.axis('off')
    layers = [LAYER0, LAYER1, LAYER2, LAYER3, LAYER4]
    for idx, layer in enumerate(layers):
        for node in layer:
            x, y = pos[node]
            box = FancyBboxPatch((x - 0.55, y - 0.28), 1.1, 0.56, boxstyle="round,pad=0.04",
                                 facecolor=COLORS[idx], edgecolor='#2C3E50', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, y, node, fontsize=9, fontweight='bold', ha='center', va='center')
    for u, v in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.annotate('', xy=(x2, y2 + 0.28), xytext=(x1, y1 - 0.28),
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='#2C3E50'))
    _add_tier_labels(ax)
    ax.set_title(
        'Tổng quan kiến trúc theo tầng\n'
        'Luồng xử lý: Dữ liệu → Tiền xử lý → Thuật toán → Huấn luyện & đánh giá',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu (matplotlib): {output_path}")


def draw_layered_architecture(output_path: str):
    try:
        draw_with_networkx(output_path)
    except Exception as e:
        print(f"NetworkX không dùng được ({e}). Dùng matplotlib.")
        draw_with_matplotlib(output_path)


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'architecture_layers.png')
    draw_layered_architecture(out_path)
    print("Mở _networkx/architecture_layers.png để xem.")
