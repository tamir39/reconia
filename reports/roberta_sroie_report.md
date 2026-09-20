# Báo Cáo Thực Nghiệm: RoBERTa trên Bộ Dữ Liệu SROIE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `RoBERTa` |
| **Bộ dữ liệu** | `SROIE` |
| **Thiết bị phần cứng** | `cuda (NVIDIA GeForce GTX 1650 Ti)` |
| **Tổng số tham số** | 129,265,929 |
| **Epoch tốt nhất** | Epoch 1 |
| **Token Macro-F1 (Tối ưu)** | **2.00%** |
| **Token Micro-F1** | **5.31%** |
| **Entity Micro-F1** | **0.00%** |
| **Độ trễ suy luận trung bình** | **58.96 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **1547.0 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![roberta_sroie_training_curves](roberta_sroie_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-address` | 0.00% | 0.00% | **0.00%** | 15 |
| `B-company` | 0.00% | 0.00% | **0.00%** | 8 |
| `B-date` | 0.00% | 0.00% | **0.00%** | 5 |
| `B-total` | 0.00% | 0.00% | **0.00%** | 5 |
| `I-address` | 25.00% | 7.89% | **12.00%** | 38 |
| `I-company` | 0.00% | 0.00% | **0.00%** | 17 |
| `I-date` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 84.36% | 96.04% | **89.83%** | 455 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 2.0562 | 1.0738 | 5.31% | 2.00% | 0.00% | 59.0 ms | 1547.0 MB |
