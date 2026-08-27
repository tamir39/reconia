# ERD — Reconia

11 bảng. Nguồn sự thật là `app/models/`; sơ đồ này để đọc và để đưa vào báo cáo.

```mermaid
erDiagram
    PARTY ||--o{ DOCUMENT : "phát hành"
    PARTY ||--o{ PRODUCT_ALIAS : "cách gọi riêng"
    PRODUCT ||--o{ PRODUCT_ALIAS : "được ánh xạ tới"
    PRODUCT ||--o{ DOCUMENT_LINE : "được phân giải thành"
    PRODUCT ||--o{ UOM_CONVERSION : "quy đổi đơn vị"

    DOCUMENT ||--o{ DOCUMENT_LINE : "gồm"
    DOCUMENT ||--o{ FIELD_EVIDENCE : "bằng chứng"
    DOCUMENT_LINE ||--o{ FIELD_EVIDENCE : "bằng chứng dòng"

    RECONCILIATION_BATCH ||--o{ RECONCILIATION_RESULT : "sinh ra"
    DOCUMENT ||--o{ RECONCILIATION_RESULT : "hóa đơn"
    DOCUMENT ||--o{ RECONCILIATION_RESULT : "phiếu kho"
    DOCUMENT ||--o{ RECONCILIATION_RESULT : "đơn đặt hàng"
    RECONCILIATION_RESULT ||--o{ FINDING : "vi phạm"
    DOCUMENT ||--o{ FINDING : "vi phạm"

    PARTY {
        int id PK
        string tax_code UK "MST — khóa định danh duy nhất"
        string name
        string name_norm
        string address "chỉ để hiển thị, KHÔNG dùng để khớp"
    }
    PRODUCT {
        int id PK
        string code UK
        string name
        string name_norm
        string base_uom
        numeric size_value "cổng thuộc tính"
        string size_uom
        numeric pack_qty
    }
    PRODUCT_ALIAS {
        int id PK
        int party_id FK
        int product_id FK
        string alias_norm "UNIQUE(party_id, alias_norm)"
        string confirmed_by "bộ nhớ hiệu chỉnh"
    }
    UOM_CONVERSION {
        int id PK
        int product_id FK "NULL = quy tắc chung"
        string from_uom
        string to_uom
        numeric factor
    }
    DOCUMENT {
        int id PK
        enum doc_type "INVOICE|STOCK_IN|STOCK_OUT|PURCHASE_ORDER"
        enum direction "IN|OUT"
        int party_id FK
        string doc_no
        string doc_serial "ký hiệu"
        date doc_date
        enum source "XML|PDF_TEXT|OCR|MANUAL"
        string file_sha256 "trùng lặp L0"
        numeric total "NUMERIC, không bao giờ float"
        enum status
        string ref_doc_no "ghép nối tầng 0"
        date ref_doc_date
    }
    DOCUMENT_LINE {
        int id PK
        int document_id FK
        int line_no
        string raw_name "giữ nguyên văn"
        string supplier_item_code "khóa khớp tốt nhất"
        int product_id FK "NULL = chưa phân giải"
        numeric quantity
        numeric base_quantity "đã quy đổi"
        numeric unit_price
        numeric line_amount
        numeric tax_rate
    }
    FIELD_EVIDENCE {
        int id PK
        int document_id FK
        int line_id FK
        string field_name
        string raw_value
        numeric confidence
        int page
        numeric bbox_x0 "lưu từ đầu — về sau rất đắt"
    }
    RECONCILIATION_BATCH {
        int id PK
        date period_start
        date period_end
        int party_id FK
        enum direction
    }
    RECONCILIATION_RESULT {
        int id PK
        int batch_id FK
        int invoice_id FK
        int stock_doc_id FK
        int po_id FK
        int product_id FK
        numeric qty_ordered "đã đặt"
        numeric qty_moved "thực nhận/xuất"
        numeric qty_invoiced "xuất hóa đơn"
        enum outcome "5 nhóm kết quả"
        enum match_tier "0|1|2"
        string reason "lý do đọc được"
        enum reviewed_outcome "người ghi đè"
    }
    FINDING {
        int id PK
        int document_id FK
        int result_id FK
        string rule_code
        enum severity
        string message "tiếng Việt"
        json details
    }
    AUDIT_EVENT {
        int id PK
        string entity_type
        int entity_id
        string action
        string actor "người dùng, hoặc 'agent'"
        string tool_name "Agent đã gọi công cụ nào"
        numeric confidence
        string reason
    }
```

## Ba quyết định thiết kế đáng chú ý

**Một bảng `document` cho cả bốn loại.** `doc_type` nói nó là gì, `direction` nói hàng đi
chiều nào. Nhờ vậy engine đối soát chạy **không đổi một dòng** cho cả chiều nhập lẫn chiều
xuất — đó chính là điều làm *"chứng từ nhập xuất"* thành một engine chứ không phải hai.

**`RECONCILIATION_RESULT` giữ cả ba số lượng.** Đặt, thực nhận, xuất hóa đơn nằm trên cùng
một dòng nên chênh lệch ba chiều nhìn thấy ngay, không cần join lại.

**`FIELD_EVIDENCE` có tọa độ ngay từ đầu.** Màn hình đối chiếu song song tốn 1 ngày nếu đã
có tọa độ, tốn hơn một tuần nếu phải làm lại từ đầu.
