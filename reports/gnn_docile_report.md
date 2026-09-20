# Báo Cáo Thực Nghiệm: GNN trên Bộ Dữ Liệu DOCILE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `GNN` |
| **Bộ dữ liệu** | `DOCILE` |
| **Thiết bị phần cứng** | `cuda` |
| **Tổng số tham số** | 3,949,297 |
| **Epoch tốt nhất** | Epoch 11 |
| **Token Macro-F1 (Tối ưu)** | **29.89%** |
| **Token Micro-F1** | **45.77%** |
| **Entity Micro-F1** | **32.09%** |
| **Độ trễ suy luận trung bình** | **7.59 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **145.2 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![gnn_docile_training_curves](gnn_docile_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-account_num` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-amount_due` | 22.22% | 37.84% | **28.00%** | 74 |
| `B-amount_paid` | 18.18% | 40.00% | **25.00%** | 5 |
| `B-amount_total_gross` | 14.18% | 28.99% | **19.05%** | 69 |
| `B-amount_total_net` | 6.25% | 5.26% | **5.71%** | 19 |
| `B-amount_total_tax` | 0.00% | 0.00% | **0.00%** | 7 |
| `B-bank_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-bic` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-currency_code_amount_due` | 12.20% | 83.33% | **21.28%** | 6 |
| `B-customer_billing_address` | 12.50% | 50.00% | **20.00%** | 4 |
| `B-customer_billing_name` | 35.10% | 54.08% | **42.57%** | 98 |
| `B-customer_delivery_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_delivery_name` | 50.00% | 50.00% | **50.00%** | 8 |
| `B-customer_id` | 20.55% | 45.45% | **28.30%** | 33 |
| `B-customer_order_id` | 3.23% | 11.11% | **5.00%** | 9 |
| `B-customer_other_address` | 50.00% | 100.00% | **66.67%** | 1 |
| `B-customer_other_name` | 31.43% | 50.00% | **38.60%** | 22 |
| `B-customer_registration_id` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-customer_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-date_due` | 26.19% | 52.38% | **34.92%** | 21 |
| `B-date_issue` | 32.11% | 80.61% | **45.93%** | 98 |
| `B-document_id` | 41.54% | 61.36% | **49.54%** | 88 |
| `B-iban` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_amount_gross` | 31.93% | 66.91% | **43.23%** | 272 |
| `B-line_item_amount_net` | 25.00% | 15.79% | **19.35%** | 76 |
| `B-line_item_code` | 27.10% | 61.70% | **37.66%** | 47 |
| `B-line_item_currency` | 15.10% | 87.88% | **25.78%** | 33 |
| `B-line_item_date` | 49.70% | 75.84% | **60.05%** | 327 |
| `B-line_item_description` | 26.29% | 46.93% | **33.70%** | 326 |
| `B-line_item_discount_amount` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-line_item_discount_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_hts_number` | 100.00% | 100.00% | **100.00%** | 1 |
| `B-line_item_order_id` | 0.00% | 0.00% | **0.00%** | 6 |
| `B-line_item_person_name` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_position` | 55.87% | 79.37% | **65.57%** | 126 |
| `B-line_item_quantity` | 27.43% | 91.56% | **42.22%** | 237 |
| `B-line_item_tax` | 50.00% | 100.00% | **66.67%** | 2 |
| `B-line_item_tax_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_unit_price_gross` | 47.19% | 55.64% | **51.07%** | 257 |
| `B-line_item_unit_price_net` | 0.00% | 0.00% | **0.00%** | 16 |
| `B-line_item_units_of_measure` | 14.29% | 9.09% | **11.11%** | 44 |
| `B-line_item_weight` | 50.00% | 100.00% | **66.67%** | 1 |
| `B-order_id` | 22.06% | 31.25% | **25.86%** | 48 |
| `B-payment_reference` | 0.00% | 0.00% | **0.00%** | 4 |
| `B-payment_terms` | 17.53% | 90.00% | **29.35%** | 30 |
| `B-tax_detail_gross` | 11.11% | 25.00% | **15.38%** | 4 |
| `B-tax_detail_net` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-tax_detail_rate` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-tax_detail_tax` | 4.00% | 33.33% | **7.14%** | 6 |
| `B-vendor_address` | 16.88% | 59.09% | **26.26%** | 22 |
| `B-vendor_email` | 38.89% | 70.00% | **50.00%** | 10 |
| `B-vendor_name` | 26.47% | 56.25% | **36.00%** | 128 |
| `B-vendor_order_id` | 15.38% | 33.33% | **21.05%** | 6 |
| `B-vendor_registration_id` | 33.33% | 100.00% | **50.00%** | 1 |
| `B-vendor_tax_id` | 71.43% | 83.33% | **76.92%** | 6 |
| `I-account_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_due` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-amount_paid` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-amount_total_gross` | 0.00% | 0.00% | **0.00%** | 6 |
| `I-amount_total_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_total_tax` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-bank_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-bic` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-currency_code_amount_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_billing_address` | 83.00% | 75.39% | **79.01%** | 764 |
| `I-customer_billing_name` | 58.31% | 60.59% | **59.42%** | 307 |
| `I-customer_delivery_address` | 54.84% | 30.91% | **39.53%** | 55 |
| `I-customer_delivery_name` | 38.46% | 41.67% | **40.00%** | 12 |
| `I-customer_id` | 100.00% | 100.00% | **100.00%** | 1 |
| `I-customer_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_other_address` | 20.59% | 38.89% | **26.92%** | 18 |
| `I-customer_other_name` | 8.97% | 31.82% | **14.00%** | 44 |
| `I-customer_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-date_due` | 25.00% | 50.00% | **33.33%** | 4 |
| `I-date_issue` | 32.80% | 85.42% | **47.40%** | 48 |
| `I-document_id` | 0.00% | 0.00% | **0.00%** | 8 |
| `I-iban` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_amount_gross` | 6.67% | 100.00% | **12.50%** | 1 |
| `I-line_item_amount_net` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-line_item_code` | 14.29% | 16.67% | **15.38%** | 6 |
| `I-line_item_currency` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_date` | 23.81% | 19.61% | **21.51%** | 51 |
| `I-line_item_description` | 22.51% | 55.71% | **32.06%** | 806 |
| `I-line_item_discount_amount` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_discount_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_hts_number` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_person_name` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_position` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_quantity` | 25.64% | 100.00% | **40.82%** | 10 |
| `I-line_item_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_tax_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_unit_price_gross` | 100.00% | 50.00% | **66.67%** | 6 |
| `I-line_item_unit_price_net` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-line_item_units_of_measure` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_weight` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-order_id` | 0.00% | 0.00% | **0.00%** | 4 |
| `I-payment_reference` | 0.00% | 0.00% | **0.00%** | 2 |
| `I-payment_terms` | 15.29% | 70.59% | **25.13%** | 68 |
| `I-tax_detail_gross` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_address` | 64.41% | 77.26% | **70.25%** | 787 |
| `I-vendor_email` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_name` | 37.82% | 60.06% | **46.42%** | 318 |
| `I-vendor_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 96.75% | 81.07% | **88.22%** | 27,472 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 3.6571 | 2.9748 | 16.36% | 3.78% | 11.02% | 106.3 ms | 145.2 MB |
| 2 | 2.5944 | 2.1541 | 31.91% | 10.78% | 22.67% | 12.7 ms | 145.2 MB |
| 3 | 1.9436 | 1.8416 | 28.68% | 12.21% | 20.59% | 11.7 ms | 145.2 MB |
| 4 | 1.5442 | 1.6592 | 33.50% | 18.10% | 22.49% | 9.1 ms | 145.2 MB |
| 5 | 1.2516 | 1.7536 | 36.89% | 19.12% | 25.27% | 9.2 ms | 145.2 MB |
| 6 | 1.0437 | 1.6926 | 37.26% | 22.47% | 25.62% | 10.2 ms | 145.2 MB |
| 7 | 0.8803 | 1.6947 | 39.94% | 25.42% | 27.45% | 9.5 ms | 145.2 MB |
| 8 | 0.7606 | 1.7833 | 41.47% | 26.88% | 28.32% | 8.6 ms | 145.2 MB |
| 9 | 0.6643 | 1.9314 | 45.28% | 27.43% | 31.46% | 7.8 ms | 145.2 MB |
| 10 | 0.5968 | 1.9427 | 44.47% | 27.80% | 30.89% | 7.3 ms | 145.2 MB |
| 11 | 0.5329 | 2.0363 | 45.77% | 29.89% | 32.09% | 7.6 ms | 145.2 MB |
| 12 | 0.4810 | 1.9742 | 43.46% | 27.58% | 30.14% | 7.6 ms | 145.2 MB |
| 13 | 0.4487 | 2.0768 | 43.66% | 27.99% | 30.35% | 7.1 ms | 145.2 MB |
| 14 | 0.4087 | 2.1407 | 44.65% | 28.27% | 31.26% | 8.6 ms | 145.2 MB |
| 15 | 0.3874 | 2.1429 | 43.77% | 27.32% | 30.30% | 6.6 ms | 145.2 MB |
| 16 | 0.3678 | 2.1644 | 45.32% | 29.08% | 31.51% | 7.0 ms | 145.2 MB |
