# Bảng So Sánh Tổng Hợp Các Mô Hình

Tổng hợp kết quả thực nghiệm từ các file trong thư mục `reports/` (ưu tiên đánh giá trên tập **TEST** độc lập):

## 1. Kết Quả Tổng Hợp

| Mô hình | Bộ dữ liệu | Tập | Epoch | Precision (%) | Recall (%) | F1 (%) | Latency (ms) |
|---|---|:---:|:---:|---|---|---|---|
| **GNN** | `CORD` | `TEST` | 19 | 64.50% | 75.89% | **69.73%** | 59.90 ms |
| **LayoutLMv3** | `CORD` | `TEST` | 20 | 93.93% | 93.93% | **93.93%** | 188.74 ms |
| **RoBERTa** | `CORD` | `TEST` | 9 | 91.71% | 92.27% | **91.99%** | 78.70 ms |
| **GNN** | `DOCILE` | `TEST` | 19 | 63.14% | 70.95% | **66.82%** | 95.11 ms |
| **RoBERTa** | `DOCILE` | `TEST` | 19 | 77.46% | 82.05% | **79.69%** | 94.56 ms |
| **GNN** | `SROIE` | `TEST` | 15 | 78.19% | 83.52% | **80.77%** | 16.34 ms |
| **LayoutLMv3** | `SROIE` | `TEST` | 9 | 93.84% | 95.17% | **94.50%** | 160.48 ms |
| **RoBERTa** | `SROIE` | `TEST` | 8 | 86.22% | 90.62% | **88.37%** | 41.81 ms |

## 2. Biểu Đồ So Sánh

![benchmark_comparison](figures/benchmark_comparison.png)
