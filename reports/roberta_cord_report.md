# Báo Cáo Thực Nghiệm: RoBERTa trên Bộ Dữ Liệu CORD

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `RoBERTa` |
| **Bộ dữ liệu** | `CORD` |
| **Thiết bị phần cứng** | `cuda (NVIDIA GeForce GTX 1650 Ti)` |
| **Tổng số tham số** | 129,304,379 |
| **Epoch tốt nhất** | Epoch 1 |
| **Token Macro-F1 (Tối ưu)** | **0.84%** |
| **Token Micro-F1** | **2.53%** |
| **Entity Micro-F1** | **3.28%** |
| **Độ trễ suy luận trung bình** | **55.87 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **1547.9 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![roberta_cord_training_curves](roberta_cord_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-menu.cnt` | 0.00% | 0.00% | **0.00%** | 7 |
| `B-menu.discountprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.itemsubtotal` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.nm` | 11.11% | 14.29% | **12.50%** | 7 |
| `B-menu.num` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.price` | 0.00% | 0.00% | **0.00%** | 7 |
| `B-menu.sub.cnt` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-menu.sub.nm` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-menu.sub.price` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-menu.sub.unitprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-menu.unitprice` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-menu.vatyn` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-sub_total.discount_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-sub_total.etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-sub_total.othersvc_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-sub_total.service_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-sub_total.subtotal_price` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-sub_total.tax_price` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-total.cashprice` | 0.00% | 0.00% | **0.00%** | 5 |
| `B-total.changeprice` | 0.00% | 0.00% | **0.00%** | 5 |
| `B-total.creditcardprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-total.emoneyprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-total.menuqty_cnt` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-total.menutype_cnt` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-total.total_etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-total.total_price` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-void_menu.nm` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-void_menu.price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.cnt` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.discountprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.itemsubtotal` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.nm` | 5.00% | 11.11% | **6.90%** | 9 |
| `I-menu.num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.sub.cnt` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.sub.nm` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-menu.sub.price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.sub.unitprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.unitprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-menu.vatyn` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-sub_total.discount_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-sub_total.etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-sub_total.othersvc_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-sub_total.service_price` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-sub_total.subtotal_price` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-sub_total.tax_price` | 0.00% | 0.00% | **0.00%** | 3 |
| `I-total.cashprice` | 0.00% | 0.00% | **0.00%** | 7 |
| `I-total.changeprice` | 0.00% | 0.00% | **0.00%** | 5 |
| `I-total.creditcardprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total.emoneyprice` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total.menuqty_cnt` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-total.menutype_cnt` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total.total_etc` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-total.total_price` | 0.00% | 0.00% | **0.00%** | 5 |
| `I-void_menu.nm` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-void_menu.price` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 0.00% | 0.00% | **0.00%** | 0 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 4.1810 | 3.8847 | 2.53% | 0.84% | 3.28% | 55.9 ms | 1547.9 MB |
