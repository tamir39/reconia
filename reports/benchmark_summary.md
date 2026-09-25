# Bảng So Sánh Tổng Hợp Các Mô Hình

Tổng hợp kết quả thực nghiệm từ các file trong thư mục `reports/` (ưu tiên đánh giá trên tập **TEST** độc lập):

Sáu lần chạy GNN/RoBERTa dưới đây đã kiểm chứng provenance và tái chấm field-value. P/R/F1 trong bảng và biểu đồ là **entity theo đoạn OCR** (`ocr-bio-exact-span-v2`), không phải field-value hoặc điểm DocILE chính thức. Xem [bảng đã kiểm chứng](verified_comparison.md) cho điểm giá trị trường, bốn key SROIE gốc và bằng chứng. Mỗi lần chạy chỉ có seed 42; độ trễ là số đo của từng phiên, chưa phải kết luận thống kê về tốc độ.

## 1. Kết Quả Tổng Hợp

| Mô hình | Bộ dữ liệu | Tập | Epoch | Precision (%) | Recall (%) | F1 (%) | Latency (ms) |
|---|---|:---:|:---:|---|---|---|---|
| **GNN** | `CORD` | `TEST` | 19 | 64.50% | 75.89% | **69.73%** | 59.90 ms |
| **RoBERTa** | `CORD` | `TEST` | 9 | 91.71% | 92.27% | **91.99%** | 78.70 ms |
| **GNN** | `DOCILE` | `TEST` | 19 | 63.14% | 70.95% | **66.82%** | 95.11 ms |
| **RoBERTa** | `DOCILE` | `TEST` | 19 | 77.46% | 82.05% | **79.69%** | 94.56 ms |
| **GNN** | `SROIE` | `TEST` | 15 | 78.19% | 83.52% | **80.77%** | 16.34 ms |
| **RoBERTa** | `SROIE` | `TEST` | 8 | 86.22% | 90.62% | **88.37%** | 41.81 ms |

## 2. Biểu Đồ So Sánh

![benchmark_comparison](figures/benchmark_comparison.png)
