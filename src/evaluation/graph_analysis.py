"""
Graph Analysis Module using NetworkX

Trực quan hóa mối quan hệ giữa các trường thông tin (Features) trong bộ dữ liệu Criteo.
Điều này giúp giải thích cấu trúc dữ liệu và các tương tác tiềm năng mà model học được.
"""
try:
    import networkx as nx
    HAS_NETWORKX = True
except (ImportError, AttributeError):
    HAS_NETWORKX = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.data.data_loader import CriteoDataLoader

class FeatureGraphAnalyzer:
    """
    Phân tích và trực quan hóa mối quan hệ giữa các Feature bằng Đồ thị.
    
    Sử dụng NetworkX để:
    1. Biểu diễn các cột dữ liệu (C1-C26, I1-I13) dưới dạng Nút (Nodes).
    2. Biểu diễn mức độ tương quan (Correlation/Association) dưới dạng Cạnh (Edges).
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        
    def calculate_interactions(self, data_path: str, max_samples: int = 5000) -> pd.DataFrame:
        """
        Tính toán ma trận tương quan giữa các cột features.
        """
        print(f"Đang tính toán tương quan đặc trưng từ {max_samples} mẫu...")
        
        loader = CriteoDataLoader(data_path, batch_size=max_samples, max_samples=max_samples)
        labels, all_features = next(iter(loader))
        
        # Chuyển sang DataFrame để dễ xử lý
        cols = [f"I{i+1}" for i in range(13)] + [f"C{i+1}" for i in range(26)]
        df = pd.DataFrame(all_features, columns=cols)
        
        # Xử lý dữ liệu số (Numerical)
        df_num = df[[f"I{i+1}" for i in range(13)]].apply(pd.to_numeric, errors='coerce').fillna(0)
        corr_matrix = df_num.corr().abs()
        
        return corr_matrix

    def visualize_feature_network(self, corr_matrix: pd.DataFrame, threshold: float = 0.3):
        """
        Vẽ đồ thị mạng lưới các đặc trưng bằng NetworkX.
        
        Args:
            corr_matrix: Ma trận tương quan
            threshold: Chỉ vẽ các cạnh có trọng số lớn hơn threshold này
        """
        if not HAS_NETWORKX:
            print("Lỗi: NetworkX không khả dụng trên phiên bản Python này (3.14+ compatibility issue).")
            print("Vui lòng bỏ qua tính năng vẽ đồ thị NetworkX và tiếp tục với Training/Evaluation.")
            return

        print(f"Đang tạo đồ thị NetworkX (threshold={threshold})...")
        
        G = nx.Graph()
        
        # Thêm các node
        for col in corr_matrix.columns:
            G.add_node(col)
            
        # Thêm các cạnh dựa trên tương quan
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                weight = corr_matrix.iloc[i, j]
                if weight > threshold:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=weight)
        
        # Vẽ đồ thị
        plt.figure(figsize=(12, 10))
        
        # Định dạng vị trí các nút (Spring layout tạo hiệu ứng vật lý đẹp)
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Vẽ các nút
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9)
        
        # Vẽ các cạnh (độ đậm nhạt dựa trên trọng số)
        edges = G.edges(data=True)
        weights = [d['weight'] * 5 for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5)
        
        # Vẽ nhãn
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')
        
        plt.title(f"Mạng lưới tương quan giữa các đặc trưng Numerical (Tương quan > {threshold})", 
                  fontsize=15, fontweight='bold')
        plt.axis('off')
        
        output_path = f"{self.output_dir}/feature_network.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu đồ thị tại: {output_path}")
        plt.show()

    def plot_label_dependency_graph(self, data_path: str, max_samples: int = 5000):
        """
        Vẽ đồ thị kết nối trực tiếp từ Label tới các Features quan trọng nhất.
        """
        # Đây là một ý tưởng hay để xem feature nào "quyết định" việc Click
        pass # Có thể mở rộng sau

if __name__ == "__main__":
    # Test nhanh
    import os
    sample_data = "/Users/coinhat/Documents/PKA/TTUD/data/sample/train.txt"
    if os.path.exists(sample_data):
        analyzer = FeatureGraphAnalyzer()
        corr = analyzer.calculate_interactions(sample_data)
        analyzer.visualize_feature_network(corr, threshold=0.1)
