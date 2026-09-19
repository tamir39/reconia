# LƯỢC ĐỒ QUAN HỆ DỮ LIỆU

**Đề tài:** Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin
từ hóa đơn (ĐA161)

> [!NOTE]
> **Trạng thái: Lớp ứng dụng mở rộng (Frozen Demo Layer).**
> Lược đồ cơ sở dữ liệu dưới đây phục vụ cho lớp ứng dụng đối soát 3 chiều và lưu vết thực thể (ERP 3-way matching demo layer) trong `app/models/`. Lớp ứng dụng này đã được đóng băng để tập trung vào mục tiêu nghiên cứu học thuật của đề tài: so sánh và đánh giá 4 mô hình Deep Learning trên 3 bộ dữ liệu hóa đơn.

Lược đồ gồm mười một bảng. Định nghĩa chính thức nằm trong `app/models/`; sơ đồ dưới đây
biểu diễn lại cấu trúc đó phục vụ việc đọc hiểu và tham khảo:

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
        string address "chỉ hiển thị, không dùng so khớp"
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
        string alias_norm "duy nhất theo party_id"
        string confirmed_by "người xác nhận"
    }
    UOM_CONVERSION {
        int id PK
        int product_id FK "rỗng: quy tắc chung"
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
        string file_sha256 "phát hiện trùng lớp L0"
        numeric total "kiểu NUMERIC"
        enum status
        string ref_doc_no "tham chiếu tầng 0"
        date ref_doc_date
    }
    DOCUMENT_LINE {
        int id PK
        int document_id FK
        int line_no
        string raw_name "giữ nguyên văn bản gốc"
        string supplier_item_code "mã hàng nhà cung cấp"
        int product_id FK "rỗng: chưa phân giải"
        numeric quantity
        numeric base_quantity "quy đổi về đơn vị cơ sở"
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
        numeric bbox_x0 "tọa độ vùng văn bản"
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
        numeric qty_ordered "số lượng đặt"
        numeric qty_moved "số lượng thực nhận hoặc xuất"
        numeric qty_invoiced "số lượng xuất hóa đơn"
        enum outcome "năm nhóm kết quả"
        enum match_tier "0|1|2"
        string reason "căn cứ kết luận"
        enum reviewed_outcome "kết quả người dùng ghi đè"
    }
    FINDING {
        int id PK
        int document_id FK
        int result_id FK
        string rule_code
        enum severity
        string message "nội dung hiển thị"
        json details
    }
    AUDIT_EVENT {
        int id PK
        string entity_type
        int entity_id
        string action
        string actor "người dùng hoặc agent"
        string tool_name "công cụ được gọi"
        numeric confidence
        string reason
    }
```

## Các quyết định thiết kế chính

**Sử dụng một bảng `DOCUMENT` cho cả bốn loại chứng từ.** Trường `doc_type` xác định loại chứng
từ, trường `direction` xác định chiều luân chuyển hàng hóa. Nhờ đó thành phần đối soát vận hành
không thay đổi đối với cả chiều nhập và chiều xuất, thay vì phải xây dựng hai thành phần riêng biệt.

**Bảng `RECONCILIATION_RESULT` lưu đồng thời ba đại lượng số lượng.** Số lượng đặt, số lượng thực
nhận hoặc xuất, và số lượng xuất hóa đơn cùng nằm trên một bản ghi, cho phép xác định chênh lệch
ba chiều mà không cần thực hiện phép kết nối bổ sung.

**Bảng `FIELD_EVIDENCE` lưu tọa độ vùng văn bản ngay từ lần trích xuất đầu tiên.** Việc bổ sung
tọa độ sau khi dữ liệu đã được trích xuất đòi hỏi xử lý lại toàn bộ tập chứng từ.
