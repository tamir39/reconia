# Báo Cáo Thực Nghiệm: RoBERTa trên Bộ Dữ Liệu DOCILE

## 1. Tóm Tắt Kết Quả Tối Ưu Nhất

| Tiêu chí đánh giá | Giá trị đạt được |
|---|---|
| **Mô hình** | `RoBERTa` |
| **Bộ dữ liệu** | `DOCILE` |
| **Thiết bị phần cứng** | `cuda (NVIDIA GeForce GTX 1650 Ti)` |
| **Tổng số tham số** | 129,344,367 |
| **Epoch tốt nhất** | Epoch 1 |
| **Token Macro-F1 (Tối ưu)** | **0.00%** |
| **Token Micro-F1** | **0.00%** |
| **Entity Micro-F1** | **0.00%** |
| **Độ trễ suy luận trung bình** | **61.98 ms / tài liệu** |
| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **1548.3 MB** |

## 2. Biểu Đồ Huấn Luyện & Đánh Giá

![roberta_docile_training_curves](roberta_docile_training_curves.png)

## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)

| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |
|---|---|---|---|---|
| `B-account_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-amount_due` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-amount_paid` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-amount_total_gross` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-amount_total_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-amount_total_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-bank_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-bic` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-currency_code_amount_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_billing_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_billing_name` | 0.00% | 0.00% | **0.00%** | 6 |
| `B-customer_delivery_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_delivery_name` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-customer_id` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-customer_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_other_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_other_name` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-customer_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-customer_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-date_due` | 0.00% | 0.00% | **0.00%** | 2 |
| `B-date_issue` | 0.00% | 0.00% | **0.00%** | 5 |
| `B-document_id` | 0.00% | 0.00% | **0.00%** | 4 |
| `B-iban` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_amount_gross` | 0.00% | 0.00% | **0.00%** | 11 |
| `B-line_item_amount_net` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-line_item_code` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-line_item_currency` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_date` | 0.00% | 0.00% | **0.00%** | 22 |
| `B-line_item_description` | 0.00% | 0.00% | **0.00%** | 9 |
| `B-line_item_discount_amount` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_discount_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_hts_number` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_person_name` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_position` | 0.00% | 0.00% | **0.00%** | 8 |
| `B-line_item_quantity` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-line_item_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_tax_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-line_item_unit_price_gross` | 0.00% | 0.00% | **0.00%** | 13 |
| `B-line_item_unit_price_net` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-line_item_units_of_measure` | 0.00% | 0.00% | **0.00%** | 7 |
| `B-line_item_weight` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-order_id` | 0.00% | 0.00% | **0.00%** | 6 |
| `B-payment_reference` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-payment_terms` | 0.00% | 0.00% | **0.00%** | 1 |
| `B-tax_detail_gross` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-tax_detail_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-tax_detail_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-tax_detail_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-vendor_address` | 0.00% | 0.00% | **0.00%** | 3 |
| `B-vendor_email` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-vendor_name` | 0.00% | 0.00% | **0.00%** | 4 |
| `B-vendor_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-vendor_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `B-vendor_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-account_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_paid` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_total_gross` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-amount_total_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-amount_total_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-bank_num` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-bic` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-currency_code_amount_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_billing_address` | 0.00% | 0.00% | **0.00%** | 71 |
| `I-customer_billing_name` | 0.00% | 0.00% | **0.00%** | 20 |
| `I-customer_delivery_address` | 0.00% | 0.00% | **0.00%** | 3 |
| `I-customer_delivery_name` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-customer_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_other_address` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_other_name` | 0.00% | 0.00% | **0.00%** | 8 |
| `I-customer_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-customer_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-date_due` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-date_issue` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-document_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-iban` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_amount_gross` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_amount_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_code` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_currency` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_date` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_description` | 0.00% | 0.00% | **0.00%** | 26 |
| `I-line_item_discount_amount` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_discount_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_hts_number` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_person_name` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_position` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_quantity` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_tax_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_unit_price_gross` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-line_item_unit_price_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_units_of_measure` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-line_item_weight` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-payment_reference` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-payment_terms` | 0.00% | 0.00% | **0.00%** | 1 |
| `I-tax_detail_gross` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_net` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_rate` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-tax_detail_tax` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_address` | 0.00% | 0.00% | **0.00%** | 40 |
| `I-vendor_email` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_name` | 0.00% | 0.00% | **0.00%** | 10 |
| `I-vendor_order_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_registration_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `I-vendor_tax_id` | 0.00% | 0.00% | **0.00%** | 0 |
| `O` | 63.12% | 100.00% | **77.39%** | 510 |

## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)

| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 1 | 4.1844 | 2.6937 | 0.00% | 0.00% | 0.00% | 62.0 ms | 1548.3 MB |
