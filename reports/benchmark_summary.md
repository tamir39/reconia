# Bảng So Sánh Tổng Hợp Các Mô Hình

> Có kết quả lịch sử chưa được tái lập bằng quy trình mới. Không dùng bảng này làm kết luận cuối cùng.

Tổng hợp kết quả thực nghiệm từ các file trong thư mục `reports/` (ưu tiên đánh giá trên tập **TEST** độc lập):

## 1. Kết Quả Tổng Hợp

| Mô hình | Bộ dữ liệu | Tập | Epoch | Precision (%) | Recall (%) | F1 (%) | Latency (ms) |
|---|---|:---:|:---:|---|---|---|---|
| **GNN** | `CORD` | `TEST` | 15 | 64.09% | 75.36% | **69.27%** | 35.98 ms |
| **RoBERTa** | `CORD` | `TEST` | 12 | 91.03% | 91.51% | **91.27%** | 577.28 ms |
| **GNN** | `DOCILE` | `TEST` | 19 | 63.14% | 70.95% | **66.82%** | 95.11 ms |
| **RoBERTa** | `DOCILE` | `TEST` | 19 | 77.46% | 82.05% | **79.69%** | 94.56 ms |
| **GNN** | `SROIE` | `TEST` | 12 | 80.54% | 84.66% | **82.55%** | 21.89 ms |
| **RoBERTa** | `SROIE` | `TEST` | 15 | 86.33% | 91.48% | **88.83%** | 622.72 ms |

## 2. Biểu Đồ So Sánh

![benchmark_comparison](figures/benchmark_comparison.png)
