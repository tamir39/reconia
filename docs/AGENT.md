# KIẾN TRÚC AGENT — Module 5

Phác thảo kiến trúc Agent điều phối pipeline. Hiện thực ở **Tuần 8**.

Theo định hướng GVHD: *"OCR nghiên cứu kỹ 40%, 60% còn lại do Agent"* và *"pipeline giữ nguyên"*.
Agent **không thay thế** 6 module — nó **điều phối** chúng và xử lý phần mà mã lệnh xác định
không giải quyết được.

---

## 1. Vì sao cần Agent

Pipeline cố định thất bại ở đúng những chỗ nghiệp vụ thật hay gặp:

| Tình huống | Pipeline cố định | Agent |
|---|---|---|
| PDF không có lớp văn bản | Trả về rỗng, dừng | Nhận ra rỗng, chuyển sang đường nhận dạng ảnh |
| Nhận dạng ra tên hàng sai chính tả | Không khớp được, bỏ qua | Thử bí danh, rồi khớp gần đúng, rồi hỏi người |
| Số học lệch 1000 lần | Báo sai số học | Suy ra là lỗi phân tích dấu phân cách, đọc lại |
| Thiếu trường bắt buộc | Đổ lỗi | Thử đường trích xuất khác trước khi bỏ cuộc |

Phần "60%" chính là những nhánh này: **quyết định làm gì khi bước trước không đạt.**

## 2. Trạng thái xử lý

Mỗi chứng từ đi qua một máy trạng thái. Trạng thái được **lưu trong CSDL**, không giữ trong
bộ nhớ, để có thể tạm dừng và chạy tiếp sau khi người dùng bổ sung thông tin.

```
NEW ──► CLASSIFYING ──► EXTRACTING ──► NORMALIZING ──► RESOLVING
                            │                              │
                            └──── retry / đổi công cụ ◄─────┘
                                           │
                                           ▼
                        MATCHING ──► VALIDATING ──► ROUTING
                                                       │
                              ┌────────────────────────┼────────────────────┐
                              ▼                        ▼                    ▼
                          APPROVED              NEEDS_REVIEW            REJECTED
                        (đủ tin cậy)          (chờ người duyệt)      (không xử lý được)
                                                       │
                                              người sửa → quay lại
                                                  NORMALIZING
```

`NEEDS_REVIEW` là điểm **tạm dừng**. Người dùng sửa xong thì luồng chạy tiếp từ đúng chỗ dừng,
không chạy lại từ đầu.

## 3. Bộ công cụ Agent được phép gọi

Agent **không** viết SQL và **không** tự do gọi hàm. Nó chỉ chọn trong một sổ đăng ký công cụ
có chữ ký rõ ràng. Đây là ranh giới an toàn, và cũng là thứ khiến hành vi của Agent ghi vết được.

| Công cụ | Đầu vào | Đầu ra | Module |
|---|---|---|---|
| `classify_document` | tệp | loại chứng từ + độ tin cậy | 1 |
| `detect_duplicate` | tệp, khóa nghiệp vụ | có trùng không, lớp nào | 1 |
| `extract_xml` | tệp XML | dữ liệu có cấu trúc | 2 |
| `extract_pdf_text` | tệp PDF | văn bản + tọa độ | 2 |
| `extract_ocr` | ảnh | văn bản + tọa độ + độ tin cậy | 2 |
| `structure_text` | văn bản thô | dữ liệu theo lược đồ | 2 |
| `normalize_fields` | dữ liệu thô | dữ liệu đã chuẩn hóa | 2 |
| `resolve_product` | tên hàng, đối tác | mã sản phẩm + độ tin cậy | 2 |
| `match_documents` | đợt đối soát | các cặp ghép + tầng + lý do | 3 |
| `validate_rules` | chứng từ / cặp ghép | danh sách phát hiện | 4 |
| `request_human_review` | lý do | tạm dừng luồng | 5 |

## 4. Định tuyến theo độ tin cậy

Sau mỗi bước, Agent quyết định một trong bốn nhánh:

| Điều kiện | Hành động |
|---|---|
| Đủ tin cậy, không có phát hiện mức ERROR | **Tự động thông qua** |
| Kết quả rỗng hoặc thiếu trường bắt buộc | **Đổi công cụ** (ví dụ PDF-văn bản → nhận dạng ảnh) |
| Sai số học có dạng đã biết (lệch ~1000 lần) | **Thử lại** với cách phân tích số khác |
| Còn lại | **Tạm dừng, chờ người duyệt** |

Số lần thử lại có giới hạn. Hết lượt thì chuyển sang chờ người duyệt, không lặp vô hạn.

> **Độ tin cậy chỉ dùng để định tuyến.** Nó không bao giờ gán kết quả đối soát —
> 5 nhóm kết quả luôn tính từ chênh lệch và độ phủ (SPEC §3).

## 5. Ghi vết

Mỗi lượt Agent gọi công cụ ghi một dòng vào bảng `audit_event`:
công cụ nào, đầu vào, đầu ra, độ tin cậy, lý do chọn nhánh.

Nhờ vậy mọi chứng từ đều trả lời được câu hỏi *"vì sao hệ thống kết luận như vậy?"* —
đây là yêu cầu bắt buộc trong nghiệp vụ kế toán, và cũng là câu GVPB hay hỏi nhất.

## 6. Ràng buộc thiết kế cho giai đoạn khóa luận

Ở Dự án CNTT chỉ có **một** Agent điều phối, có khâu duyệt của con người.
Ở khóa luận sẽ là **nhiều** Agent, mỗi Agent một skill, và **bỏ** khâu duyệt của con người.

Để bước chuyển đó là *thêm Agent* chứ không phải *viết lại*, ngay từ Tuần 8 phải giữ ba điều:

1. **Sổ đăng ký công cụ tách rời khỏi bộ điều phối.** Thêm Agent = thêm một bộ điều phối
   dùng chung sổ công cụ, không sửa vào công cụ.
2. **Trạng thái nằm trong CSDL, không nằm trong bộ nhớ.** Nhiều Agent phải cùng đọc được
   một trạng thái.
3. **Mọi quyết định đều ghi kèm lý do và độ tin cậy.** Ở khóa luận, cơ chế kiểm chứng chéo
   giữa các Agent cần chính những dữ liệu này để phát hiện bất đồng.

Ngoài ra, **mọi lần người dùng ghi đè kết quả trong Dự án CNTT đều phải lưu lại** —
đó chính là tập đánh giá để đo hệ tự động hoàn toàn ở giai đoạn khóa luận.
