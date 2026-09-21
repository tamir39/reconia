# Bảng So Sánh Tổng Hợp Các Mô Hình

Tổng hợp kết quả thực nghiệm từ các file trong thư mục `reports/` (ưu tiên đánh giá trên tập **TEST** độc lập):

## 1. Kết Quả Tổng Hợp

| Mô hình | Bộ dữ liệu | Tập | Epoch | Precision (%) | Recall (%) | F1 (%) | Latency (ms) |
|---|---|:---:|:---:|---|---|---|---|
| **GNN** | `CORD` | `TEST` | 19 | 66.71% | 77.79% | **71.82%** | 36.04 ms |
| **GNN** | `DOCILE` | `TEST` | 20 | 32.55% | 62.57% | **42.82%** | 138.98 ms |
| **GNN** | `SROIE` | `TEST` | 18 | 65.31% | 91.48% | **76.21%** | 19.67 ms |

## 2. Biểu Đồ So Sánh

![benchmark_comparison](figures/benchmark_comparison.png)
