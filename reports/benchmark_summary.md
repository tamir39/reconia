# Bảng So Sánh Tổng Hợp Các Mô Hình Học Sâu

Báo cáo tự động tổng hợp từ các kết quả thực nghiệm trong thư mục `reports/`.

## 1. Bảng So Sánh Toàn Diện (Comprehensive Benchmark Table)

| Mô hình | Bộ dữ liệu | Epoch tối ưu | Token Macro-F1 | Token Micro-F1 | Entity Micro-F1 | Độ trễ (ms/doc) | GPU VRAM (MB) |
|---|---|---|---|---|---|---|---|
| **GNN** | `CORD` | Epoch 15 | **65.19%** | 84.08% | **73.31%** | 4.40 ms | 63.7 MB |
| **RoBERTa** | `CORD` | Epoch 1 | **0.84%** | 2.53% | **3.28%** | 55.87 ms | 1547.9 MB |
| **GNN** | `DOCILE` | Epoch 11 | **29.89%** | 45.77% | **32.09%** | 7.59 ms | 145.2 MB |
| **RoBERTa** | `DOCILE` | Epoch 1 | **0.00%** | 0.00% | **0.00%** | 61.98 ms | 1548.3 MB |
| **GNN** | `SROIE` | Epoch 7 | **82.27%** | 88.62% | **72.87%** | 4.78 ms | 68.6 MB |
| **RoBERTa** | `SROIE` | Epoch 1 | **2.00%** | 5.31% | **0.00%** | 58.96 ms | 1547.0 MB |

## 2. Biểu Đồ Trực Quan So Sánh (Comparative Charts)

![benchmark_comparison](figures/benchmark_comparison.png)
