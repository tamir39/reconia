# Báo Cáo Thực Nghiệm: GNN trên Bộ Dữ Liệu DOCILE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `GNN` |
| **Bộ dữ liệu** | `DOCILE` |
| **Thiết bị phần cứng** | `cuda` |
| **Tổng số tham số** | 3,949,297 |
| **Epoch tốt nhất** | Epoch 5 |
| **Token Macro-F1 (Tối ưu)** | **10.60%** |
| **Token Micro-F1** | **25.02%** |
| **Entity Micro-F1** | **17.27%** |
| **Độ trễ suy luận trung bình** | **6.40 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **145.2 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![gnn_docile_training_curves](gnn_docile_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-account_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-amount_due` | 11.11% | 5.13% | **7.02%** | 39 |
| `B-amount_paid` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-amount_total_gross` | 18.75% | 17.14% | **17.91%** | 35 |
| `B-amount_total_net` | 0.00% | 0.00% | **0.00%** | 9 |
| `B-amount_total_tax` | 0.00% | 0.00% | **0.00%** | 4 |
| `B-bank_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-bic` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-currency_code_amount_due` | 50.00% | 33.33% | **40.00%** | 3 |
| `B-customer_billing_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_billing_name` | 25.00% | 1.92% | **3.57%** | 52 |
| `B-customer_delivery_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_delivery_name` | 0.00% | 0.00% | **0.00%** | 8 |
| `B-customer_id` | 0.00% | 0.00% | **0.00%** | 19 |
| `B-customer_order_id` | 0.00% | 0.00% | **0.00%** | 8 |
| `B-customer_other_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_other_name` | 0.00% | 0.00% | **0.00%** | 9 |
| `B-customer_registration_id` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-customer_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-date_due` | 0.00% | 0.00% | **0.00%** | 9 |
| `B-date_issue` | 8.83% | 62.00% | **15.46%** | 50 |
| `B-document_id` | 16.09% | 32.56% | **21.54%** | 43 |
| `B-iban` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_amount_gross` | 42.04% | 62.50% | **50.26%** | 152 |
| `B-line_item_amount_net` | 0.00% | 0.00% | **0.00%** | 27 |
| `B-line_item_code` | 0.00% | 0.00% | **0.00%** | 31 |
| `B-line_item_currency` | 45.16% | 77.78% | **57.14%** | 18 |
| `B-line_item_date` | 33.97% | 95.27% | **50.08%** | 169 |
| `B-line_item_description` | 14.06% | 4.92% | **7.29%** | 183 |
| `B-line_item_discount_amount` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_discount_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_hts_number` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_order_id` | 0.00% | 0.00% | **0.00%** | 5 |
| `B-line_item_person_name` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_position` | 32.46% | 87.32% | **47.33%** | 71 |
| `B-line_item_quantity` | 11.04% | 72.34% | **19.15%** | 94 |
| `B-line_item_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_tax_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_unit_price_gross` | 34.81% | 92.44% | **50.57%** | 119 |
| `B-line_item_unit_price_net` | 0.00% | 0.00% | **0.00%** | 9 |
| `B-line_item_units_of_measure` | 0.00% | 0.00% | **0.00%** | 30 |
| `B-line_item_weight` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-order_id` | 0.00% | 0.00% | **0.00%** | 26 |
| `B-payment_reference` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-payment_terms` | 66.67% | 37.50% | **48.00%** | 16 |
| `B-tax_detail_gross` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-tax_detail_net` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-tax_detail_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-tax_detail_tax` | 0.00% | 0.00% | **0.00%** | 4 |
| `B-vendor_address` | 100.00% | 8.33% | **15.38%** | 12 |
| `B-vendor_email` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-vendor_name` | 1.47% | 3.03% | **1.98%** | 66 |
| `B-vendor_order_id` | 0.00% | 0.00% | **0.00%** | 5 |
| `B-vendor_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-vendor_tax_id` | 0.00% | 0.00% | **0.00%** | 3 |
| `I-account_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_due` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-amount_paid` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_total_gross` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-amount_total_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_total_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-bank_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-bic` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-currency_code_amount_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_billing_address` | 29.01% | 74.65% | **41.78%** | 434 |
| `I-customer_billing_name` | 13.62% | 55.03% | **21.83%** | 169 |
| `I-customer_delivery_address` | 0.00% | 0.00% | **0.00%** | 55 |
| `I-customer_delivery_name` | 0.00% | 0.00% | **0.00%** | 12 |
| `I-customer_id` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-customer_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_other_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_other_name` | 0.00% | 0.00% | **0.00%** | 21 |
| `I-customer_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-date_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-date_issue` | 16.20% | 100.00% | **27.88%** | 23 |
| `I-document_id` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-iban` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_amount_gross` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-line_item_amount_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_code` | 0.00% | 0.00% | **0.00%** | 6 |
| `I-line_item_currency` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_date` | 0.00% | 0.00% | **0.00%** | 9 |
| `I-line_item_description` | 8.27% | 40.40% | **13.73%** | 505 |
| `I-line_item_discount_amount` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_discount_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_hts_number` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_person_name` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_position` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_quantity` | 0.00% | 0.00% | **0.00%** | 10 |
| `I-line_item_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_tax_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_unit_price_gross` | 0.00% | 0.00% | **0.00%** | 5 |
| `I-line_item_unit_price_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_units_of_measure` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_weight` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-order_id` | 0.00% | 0.00% | **0.00%** | 4 |
| `I-payment_reference` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-payment_terms` | 25.64% | 28.57% | **27.03%** | 35 |
| `I-tax_detail_gross` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_address` | 24.82% | 31.34% | **27.70%** | 434 |
| `I-vendor_email` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_name` | 8.74% | 25.41% | **13.00%** | 185 |
| `I-vendor_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 95.36% | 59.70% | **73.43%** | 12,948 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 4.0865 | 3.6459 | 9.55% | 1.04% | 3.68% | 143.2 ms | 145.2 MB |
| 2 | 3.4155 | 3.2392 | 14.67% | 4.96% | 12.32% | 10.5 ms | 145.2 MB |
| 3 | 2.9942 | 2.9284 | 20.65% | 7.44% | 13.98% | 6.3 ms | 145.2 MB |
| 4 | 2.7128 | 2.7744 | 24.03% | 9.49% | 17.10% | 6.2 ms | 145.2 MB |
| 5 | 2.5525 | 2.7096 | 25.02% | 10.60% | 17.27% | 6.4 ms | 145.2 MB |
