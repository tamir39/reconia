# Báo Cáo Thực Nghiệm: GNN trên Bộ Dữ Liệu CORD

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `GNN` |
| **Bộ dữ liệu** | `CORD` |
| **Thiết bị phần cứng** | `cuda` |
| **Tổng số tham số** | 3,942,589 |
| **Epoch tốt nhất** | Epoch 5 |
| **Token Macro-F1 (Tối ưu)** | **54.10%** |
| **Token Micro-F1** | **78.41%** |
| **Entity Micro-F1** | **64.00%** |
| **Độ trễ suy luận trung bình** | **4.16 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **63.7 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![gnn_cord_training_curves](gnn_cord_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-menu.cnt` | 87.21% | 90.95% | **89.04%** | 210 |
| `B-menu.discountprice` | 100.00% | 33.33% | **50.00%** | 3 |
| `B-menu.etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.itemsubtotal` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.nm` | 85.78% | 90.05% | **87.86%** | 221 |
| `B-menu.num` | 36.36% | 100.00% | **53.33%** | 4 |
| `B-menu.price` | 79.36% | 78.64% | **79.00%** | 220 |
| `B-menu.sub.cnt` | 62.50% | 46.88% | **53.57%** | 32 |
| `B-menu.sub.nm` | 78.72% | 75.51% | **77.08%** | 49 |
| `B-menu.sub.price` | 33.33% | 57.14% | **42.11%** | 14 |
| `B-menu.sub.unitprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.unitprice` | 88.64% | 78.00% | **82.98%** | 50 |
| `B-menu.vatyn` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-sub_total.discount_price` | 42.86% | 75.00% | **54.55%** | 4 |
| `B-sub_total.etc` | 50.00% | 50.00% | **50.00%** | 2 |
| `B-sub_total.othersvc_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-sub_total.service_price` | 90.91% | 83.33% | **86.96%** | 12 |
| `B-sub_total.subtotal_price` | 90.77% | 90.77% | **90.77%** | 65 |
| `B-sub_total.tax_price` | 89.58% | 95.56% | **92.47%** | 45 |
| `B-total.cashprice` | 86.57% | 90.62% | **88.55%** | 64 |
| `B-total.changeprice` | 89.55% | 93.75% | **91.60%** | 64 |
| `B-total.creditcardprice` | 70.59% | 80.00% | **75.00%** | 15 |
| `B-total.emoneyprice` | 75.00% | 75.00% | **75.00%** | 4 |
| `B-total.menuqty_cnt` | 69.57% | 61.54% | **65.31%** | 26 |
| `B-total.menutype_cnt` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-total.total_etc` | 0.00% | 0.00% | **0.00%** | 4 |
| `B-total.total_price` | 88.35% | 97.85% | **92.86%** | 93 |
| `B-void_menu.nm` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-void_menu.price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.cnt` | 100.00% | 100.00% | **100.00%** | 4 |
| `I-menu.discountprice` | 61.54% | 53.33% | **57.14%** | 15 |
| `I-menu.etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.itemsubtotal` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.nm` | 86.26% | 87.80% | **87.02%** | 336 |
| `I-menu.num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.price` | 20.00% | 25.00% | **22.22%** | 4 |
| `I-menu.sub.cnt` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.sub.nm` | 71.62% | 66.25% | **68.83%** | 80 |
| `I-menu.sub.price` | 0.00% | 0.00% | **0.00%** | 3 |
| `I-menu.sub.unitprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.unitprice` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-menu.vatyn` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-sub_total.discount_price` | 0.00% | 0.00% | **0.00%** | 7 |
| `I-sub_total.etc` | 33.33% | 33.33% | **33.33%** | 3 |
| `I-sub_total.othersvc_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-sub_total.service_price` | 73.68% | 51.85% | **60.87%** | 27 |
| `I-sub_total.subtotal_price` | 55.41% | 47.13% | **50.93%** | 87 |
| `I-sub_total.tax_price` | 82.43% | 70.93% | **76.25%** | 86 |
| `I-total.cashprice` | 69.44% | 64.10% | **66.67%** | 78 |
| `I-total.changeprice` | 65.48% | 77.46% | **70.97%** | 71 |
| `I-total.creditcardprice` | 47.06% | 42.11% | **44.44%** | 19 |
| `I-total.emoneyprice` | 14.29% | 42.86% | **21.43%** | 7 |
| `I-total.menuqty_cnt` | 67.74% | 87.50% | **76.36%** | 24 |
| `I-total.menutype_cnt` | 0.00% | 0.00% | **0.00%** | 5 |
| `I-total.total_etc` | 0.00% | 0.00% | **0.00%** | 8 |
| `I-total.total_price` | 64.41% | 67.26% | **65.80%** | 113 |
| `I-void_menu.nm` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-void_menu.price` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 0.00% | 0.00% | **0.00%** | 0 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 2.7058 | 1.6943 | 61.44% | 30.95% | 46.12% | 7.1 ms | 63.7 MB |
| 2 | 1.4837 | 1.2576 | 72.60% | 45.13% | 58.26% | 8.9 ms | 63.7 MB |
| 3 | 1.0789 | 1.0722 | 75.43% | 50.28% | 59.93% | 4.1 ms | 63.7 MB |
| 4 | 0.8780 | 1.0449 | 76.62% | 53.27% | 61.68% | 4.1 ms | 63.7 MB |
| 5 | 0.7778 | 1.0057 | 78.41% | 54.10% | 64.00% | 4.2 ms | 63.7 MB |
