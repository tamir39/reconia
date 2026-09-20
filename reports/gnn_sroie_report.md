# Báo Cáo Thực Nghiệm: GNN trên Bộ Dữ Liệu SROIE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `GNN` |
| **Bộ dữ liệu** | `SROIE` |
| **Thiết bị phần cứng** | `cuda` |
| **Tổng số tham số** | 3,936,139 |
| **Epoch tốt nhất** | Epoch 7 |
| **Token Macro-F1 (Tối ưu)** | **82.27%** |
| **Token Micro-F1** | **88.62%** |
| **Entity Micro-F1** | **72.87%** |
| **Độ trễ suy luận trung bình** | **4.78 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **68.6 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![gnn_sroie_training_curves](gnn_sroie_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-address` | 94.77% | 92.61% | **93.68%** | 176 |
| `B-company` | 81.01% | 79.01% | **80.00%** | 81 |
| `B-date` | 80.52% | 100.00% | **89.21%** | 62 |
| `B-total` | 27.53% | 80.33% | **41.00%** | 61 |
| `I-address` | 94.53% | 98.53% | **96.49%** | 544 |
| `I-company` | 92.83% | 93.62% | **93.22%** | 235 |
| `I-date` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 99.16% | 96.64% | **97.88%** | 5,957 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 1.0405 | 0.3137 | 66.04% | 62.40% | 36.32% | 5.2 ms | 68.6 MB |
| 2 | 0.2392 | 0.1862 | 77.83% | 73.48% | 53.85% | 5.6 ms | 68.6 MB |
| 3 | 0.1287 | 0.2114 | 84.44% | 80.21% | 63.72% | 4.7 ms | 68.6 MB |
| 4 | 0.0896 | 0.2191 | 85.66% | 80.00% | 67.32% | 4.6 ms | 68.6 MB |
| 5 | 0.0722 | 0.1829 | 82.60% | 76.17% | 61.32% | 4.7 ms | 68.6 MB |
| 6 | 0.0552 | 0.1965 | 86.61% | 80.61% | 68.57% | 5.2 ms | 68.6 MB |
| 7 | 0.0478 | 0.2647 | 88.62% | 82.27% | 72.87% | 4.8 ms | 68.6 MB |
| 8 | 0.0396 | 0.2914 | 86.51% | 79.10% | 68.71% | 4.9 ms | 68.6 MB |
| 9 | 0.0330 | 0.2952 | 88.01% | 82.26% | 71.41% | 4.5 ms | 68.6 MB |
| 10 | 0.0324 | 0.3274 | 88.06% | 82.04% | 72.61% | 4.7 ms | 68.6 MB |
| 11 | 0.0266 | 0.2780 | 87.58% | 81.17% | 71.06% | 4.9 ms | 68.6 MB |
| 12 | 0.0230 | 0.3446 | 88.10% | 80.75% | 71.93% | 4.9 ms | 68.6 MB |
