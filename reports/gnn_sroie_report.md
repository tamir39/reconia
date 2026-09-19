# Báo Cáo Thực Nghiệm: GNN trên Bộ Dữ Liệu SROIE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `GNN` |
| **Bộ dữ liệu** | `SROIE` |
| **Thiết bị phần cứng** | `cuda` |
| **Tổng số tham số** | 3,936,139 |
| **Epoch tốt nhất** | Epoch 4 |
| **Token Macro-F1 (Tối ưu)** | **64.27%** |
| **Token Micro-F1** | **69.20%** |
| **Entity Micro-F1** | **38.70%** |
| **Độ trễ suy luận trung bình** | **4.81 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **68.6 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![gnn_sroie_training_curves](gnn_sroie_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-address` | 65.14% | 79.78% | **71.72%** | 89 |
| `B-company` | 61.22% | 73.17% | **66.67%** | 41 |
| `B-date` | 45.31% | 100.00% | **62.37%** | 29 |
| `B-total` | 11.76% | 82.76% | **20.60%** | 29 |
| `I-address` | 82.93% | 91.89% | **87.18%** | 259 |
| `I-company` | 66.91% | 91.00% | **77.12%** | 100 |
| `I-date` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 98.86% | 87.72% | **92.96%** | 2,680 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 1.7278 | 1.1655 | 36.72% | 30.29% | 12.33% | 4.7 ms | 68.6 MB |
| 2 | 0.8831 | 0.5933 | 60.37% | 53.39% | 26.72% | 5.2 ms | 68.6 MB |
| 3 | 0.4298 | 0.4005 | 60.38% | 57.30% | 30.34% | 4.7 ms | 68.6 MB |
| 4 | 0.2552 | 0.3702 | 69.20% | 64.27% | 38.70% | 4.8 ms | 68.6 MB |
| 5 | 0.1949 | 0.3495 | 69.30% | 63.42% | 38.98% | 5.1 ms | 68.6 MB |
