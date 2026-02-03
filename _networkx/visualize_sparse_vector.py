"""
Minh họa vector thưa (output cuối của Feature Hashing) bằng NetworkX.

Chạy từ thư mục gốc dự án (cần: pip install matplotlib networkx):
    python _networkx/visualize_sparse_vector.py

Hoặc:
    cd _networkx && python visualize_sparse_vector.py

Ảnh lưu tại: _networkx/sparse_vector.png và _networkx/sparse_vector_real.png
"""
import sys
import os

# Thêm root project vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')  # Không cần display để lưu file
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

import numpy as np

# Tạo vector thưa mẫu (giống output Preprocessor sau Feature Hashing)
# Format: Dict[bucket_id, value]
def get_sample_sparse_vector():
    """Trả về một vector thưa mẫu để minh họa."""
    return {
        218608: 1.0,    # Bias
        153521: 0.693,  # I1
        195592: 0.0,    # I2
        175177: 1.792,  # I3
        50667: 7.232,   # I6
        44102: 1.386,   # I7
        89201: -1.0,    # C1
        200441: 1.0,    # C2
        102333: 1.0,    # C3
        145672: -1.0,   # C4
        88123: 1.0,     # C5
    }


def _visualize_network_style_matplotlib(sparse_vec: dict, output_path: str):
    """Vẽ dạng mạng (nút trung tâm + nút bucket) bằng matplotlib thuần."""
    sparse_vec = {k: v for k, v in sparse_vec.items() if v != 0}
    n = len(sparse_vec)
    if n == 0:
        return
    # Tọa độ: center (0,0), các bucket trên vòng tròn
    cx, cy = 0.0, 0.0
    angles = [2 * np.pi * i / n for i in range(n)]
    r = 1.8
    xs = [cx] + [r * np.cos(a) for a in angles]
    ys = [cy] + [r * np.sin(a) for a in angles]
    buckets = list(sparse_vec.keys())
    values = list(sparse_vec.values())
    colors = ['gold'] + ['green' if v >= 0 else 'coral' for v in values]
    sizes = [2000] + [400 + 200 * min(abs(v), 3) for v in values]
    fig, ax = plt.subplots(figsize=(10, 10))
    # Cạnh: center -> từng bucket
    for i in range(n):
        ax.plot([cx, xs[1 + i]], [cy, ys[1 + i]], 'gray', lw=2, alpha=0.6)
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.9, edgecolors='black', linewidths=0.5)
    ax.text(cx, cy, "Vector thưa", ha='center', va='center', fontsize=10, fontweight='bold')
    for i in range(n):
        ax.text(xs[1 + i], ys[1 + i], f"{buckets[i]}\n{values[i]:.2f}", ha='center', va='center', fontsize=7)
    ax.set_title("Minh họa vector thưa (output Feature Hashing)\nChỉ các bucket có giá trị ≠ 0", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu (dạng mạng matplotlib): {output_path}")


def _visualize_with_matplotlib_only(sparse_vec: dict, output_path: str):
    """Vẽ bằng matplotlib: ưu tiên dạng mạng, không cần NetworkX."""
    _visualize_network_style_matplotlib(sparse_vec, output_path)


def visualize_with_networkx(sparse_vec: dict, output_path: str = None):
    """Vẽ đồ thị minh họa vector thưa. Ưu tiên NetworkX; fallback: dạng mạng bằng matplotlib."""
    if not HAS_MATPLOTLIB:
        print("Cần cài matplotlib: pip install matplotlib")
        return
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sparse_vector.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Bỏ các value = 0
    sparse_vec = {k: v for k, v in sparse_vec.items() if v != 0}

    try:
        import networkx as nx
    except (ImportError, AttributeError) as e:
        print(f"NetworkX không dùng được ({e}). Dùng dạng mạng matplotlib.")
        _visualize_with_matplotlib_only(sparse_vec, output_path)
        return

    G = nx.Graph()
    center = "Vector thưa"
    G.add_node(center)
    for bucket_id, value in sparse_vec.items():
        node_name = f"B{bucket_id}"
        G.add_node(node_name)
        G.add_edge(center, node_name, weight=value)

    pos = {center: (0, 0)}
    n = len(sparse_vec)
    for i, (bucket_id, _) in enumerate(sparse_vec.items()):
        angle = 2 * np.pi * i / n
        r = 1.8
        pos[f"B{bucket_id}"] = (r * np.cos(angle), r * np.sin(angle))

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=[center], node_size=2000, node_color='gold', alpha=0.9)
    other_nodes = [n for n in G.nodes() if n != center]
    colors = ['green' if G[center][n]['weight'] >= 0 else 'coral' for n in other_nodes]
    sizes = [800 + 400 * min(abs(G[center][n]['weight']), 3) for n in other_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=sizes, node_color=colors, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', alpha=0.6)
    labels = {n: (n if n == center else f"{n[1:]}\n{G[center][n]['weight']:.2f}") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
    plt.title("Minh họa vector thưa (output Feature Hashing)\nChỉ các bucket có giá trị ≠ 0", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Đã lưu (NetworkX): {output_path}")
    try:
        plt.show()
    except Exception:
        plt.close()


def visualize_from_real_data(data_path: str = None, max_samples: int = 1):
    """Lấy 1 mẫu thật từ file, qua Preprocessor, rồi vẽ vector thưa."""
    try:
        from src.data.data_loader import StreamingIterator, create_sample_data
        from src.data.preprocessing import Preprocessor
    except Exception as e:
        print(f"Không load được module: {e}")
        return

    if data_path is None or not os.path.exists(data_path):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample")
        data_path = os.path.join(data_dir, "train.txt")
        if not os.path.exists(data_path):
            os.makedirs(data_dir, exist_ok=True)
            create_sample_data(data_path, num_samples=100)
            print(f"Đã tạo dữ liệu mẫu: {data_path}")

    preprocessor = Preprocessor(num_buckets=2**18)
    it = StreamingIterator(data_path, max_samples=max_samples)
    label, raw_features = next(iter(it))
    sparse = preprocessor.transform(raw_features)
    print(f"Vector thưa từ 1 mẫu thật: {len(sparse)} bucket khác 0")
    out = os.path.join(os.path.dirname(__file__), "sparse_vector_real.png")
    visualize_with_networkx(sparse, output_path=out)


if __name__ == "__main__":
    print("1. Minh họa vector thưa mẫu (số liệu giả định)...")
    sparse_sample = get_sample_sparse_vector()
    visualize_with_networkx(sparse_sample)

    print("\n2. Minh họa vector thưa từ 1 mẫu thật (nếu có data)...")
    root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root, "data", "sample", "train.txt")
    if os.path.exists(data_path):
        visualize_from_real_data(data_path)
    else:
        print("   (Bỏ qua: chưa có data/sample/train.txt)")
