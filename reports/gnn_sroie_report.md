# Báo Cáo Thực Nghiệm: GNN trên Bộ Dữ Liệu SROIE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `GNN` |
| **Bộ dữ liệu** | `SROIE` |
| **Thiết bị phần cứng** | `cuda` |
| **Tổng số tham số** | 3,936,139 |
| **Epoch tốt nhất** | Epoch 5 |
| **Token Macro-F1 (Tối ưu)** | **80.21%** |
| **Token Micro-F1** | **85.54%** |
| **Entity Micro-F1** | **66.73%** |
| **Độ trễ suy luận trung bình** | **4.76 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **68.6 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![gnn_sroie_training_curves](gnn_sroie_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-address` | 91.21% | 94.32% | **92.74%** | 176 |
| `B-company` | 75.00% | 85.19% | **79.77%** | 81 |
| `B-date` | 75.61% | 100.00% | **86.11%** | 62 |
| `B-total` | 21.30% | 96.72% | **34.91%** | 61 |
| `I-address` | 96.24% | 98.71% | **97.46%** | 544 |
| `I-company` | 86.38% | 94.47% | **90.24%** | 235 |
| `I-date` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 99.51% | 94.68% | **97.03%** | 5,957 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 1.0349 | 0.3075 | 67.26% | 64.54% | 37.44% | 5.5 ms | 68.6 MB |
| 2 | 0.2463 | 0.1920 | 76.66% | 73.12% | 52.22% | 4.6 ms | 68.6 MB |
| 3 | 0.1320 | 0.1704 | 84.37% | 79.63% | 64.29% | 4.7 ms | 68.6 MB |
| 4 | 0.0915 | 0.1779 | 84.86% | 79.77% | 64.72% | 4.8 ms | 68.6 MB |
| 5 | 0.0760 | 0.1680 | 85.54% | 80.21% | 66.73% | 4.8 ms | 68.6 MB |
