# KIẾN TRÚC AGENT ĐIỀU PHỐI (MODULE 5)

**Đề tài:** Xây dựng hệ thống đối soát hóa đơn và chứng từ nhập xuất (ĐA161)

Tài liệu này phác thảo kiến trúc thành phần điều phối của hệ thống. Phần hiện thực được thực
hiện ở tuần 8 theo kế hoạch.

Agent không thay thế sáu module xử lý. Agent giữ vai trò điều phối các module theo trạng thái và
xử lý những tình huống mà luồng xử lý cố định không giải quyết được.

---

## 1. Cơ sở đề xuất

Luồng xử lý cố định không xử lý được một số tình huống phát sinh thường xuyên trong thực tế:

| Tình huống | Luồng cố định | Agent điều phối |
|---|---|---|
| Tệp PDF không có lớp văn bản | Trả về kết quả rỗng và kết thúc | Xác định kết quả rỗng, chuyển sang đường nhận dạng ảnh |
| Tên hàng bị nhận dạng sai chính tả | Không tìm được kết quả khớp | Lần lượt thử bảng bí danh, khớp gần đúng, sau đó chuyển người dùng xác nhận |
| Sai lệch số học theo bậc nghìn | Ghi nhận vi phạm quy tắc số học | Suy luận nguyên nhân là diễn giải sai dấu phân cách và thực hiện phân tích lại |
| Thiếu trường dữ liệu bắt buộc | Kết thúc với lỗi | Thử đường trích xuất thay thế trước khi kết thúc |

Đây là nhóm quyết định về hành động tiếp theo khi một bước xử lý không đạt yêu cầu.

## 2. Máy trạng thái

Mỗi chứng từ được xử lý theo một máy trạng thái. Trạng thái được lưu trong cơ sở dữ liệu thay vì
trong bộ nhớ tiến trình, cho phép tạm dừng luồng xử lý và tiếp tục sau khi người dùng bổ sung
thông tin.

```
NEW ──► CLASSIFYING ──► EXTRACTING ──► NORMALIZING ──► RESOLVING
                            │                              │
                            └──── thử lại / đổi công cụ ◄───┘
                                           │
                                           ▼
                        MATCHING ──► VALIDATING ──► ROUTING
                                                       │
                              ┌────────────────────────┼────────────────────┐
                              ▼                        ▼                    ▼
                          APPROVED              NEEDS_REVIEW            REJECTED
                     (đủ độ tin cậy)      (chờ người dùng duyệt)   (không xử lý được)
                                                       │
                                        người dùng hiệu chỉnh, quay lại
                                                  NORMALIZING
```

Trạng thái `NEEDS_REVIEW` là điểm tạm dừng. Sau khi người dùng hoàn tất hiệu chỉnh, luồng xử lý
tiếp tục từ đúng vị trí đã dừng thay vì thực hiện lại từ đầu.

## 3. Bộ công cụ

Agent chỉ được phép gọi các công cụ đã đăng ký trong sổ đăng ký công cụ, với chữ ký đầu vào và
đầu ra xác định. Agent không truy vấn trực tiếp cơ sở dữ liệu. Ràng buộc này bảo đảm phạm vi tác
động của Agent là xác định và mọi hành vi đều ghi vết được.

| Công cụ | Đầu vào | Đầu ra | Module |
|---|---|---|---|
| `classify_document` | Tệp chứng từ | Loại chứng từ, độ tin cậy | 1 |
| `detect_duplicate` | Tệp, khóa nghiệp vụ | Trạng thái trùng lặp, lớp phát hiện | 1 |
| `extract_xml` | Tệp XML | Dữ liệu có cấu trúc | 2 |
| `extract_pdf_text` | Tệp PDF | Văn bản, tọa độ | 2 |
| `extract_ocr` | Ảnh | Văn bản, tọa độ, độ tin cậy | 2 |
| `structure_text` | Văn bản thô | Dữ liệu theo lược đồ | 2 |
| `normalize_fields` | Dữ liệu thô | Dữ liệu đã chuẩn hóa | 2 |
| `resolve_product` | Tên hàng, đối tác | Mã sản phẩm, độ tin cậy | 2 |
| `match_documents` | Đợt đối soát | Danh sách cặp ghép, tầng, căn cứ | 3 |
| `validate_rules` | Chứng từ hoặc cặp ghép | Danh sách phát hiện | 4 |
| `request_human_review` | Lý do | Tạm dừng luồng xử lý | 5 |

## 4. Định tuyến theo độ tin cậy

Sau mỗi bước xử lý, Agent lựa chọn một trong bốn hướng xử lý tiếp theo:

| Điều kiện | Hành động |
|---|---|
| Độ tin cậy đạt ngưỡng và không có phát hiện ở mức ERROR | Tự động thông qua |
| Kết quả rỗng hoặc thiếu trường bắt buộc | Chuyển sang công cụ thay thế |
| Sai lệch số học thuộc dạng đã xác định | Thực hiện lại với phương án phân tích khác |
| Các trường hợp còn lại | Tạm dừng, chuyển hàng đợi rà soát |

Số lần thử lại được giới hạn. Khi vượt quá giới hạn, chứng từ được chuyển sang hàng đợi rà soát.

Độ tin cậy chỉ được sử dụng cho mục đích định tuyến. Năm nhóm kết quả đối soát luôn được tính từ
chênh lệch định lượng và độ phủ, theo mục 3 của tài liệu đặc tả.

## 5. Cơ chế ghi vết

Mỗi lượt gọi công cụ được ghi một bản ghi vào bảng `audit_event`, bao gồm: tên công cụ, dữ liệu
đầu vào, dữ liệu đầu ra, độ tin cậy và căn cứ lựa chọn hướng xử lý.

Cơ chế này bảo đảm mọi kết luận của hệ thống đều truy vết được nguyên nhân, đáp ứng yêu cầu về
khả năng giải thích trong nghiệp vụ kế toán.

## 6. Ràng buộc thiết kế hướng tới giai đoạn khóa luận

Trong phạm vi Dự án CNTT, hệ thống sử dụng một Agent điều phối kết hợp với khâu rà soát của người
dùng. Ở giai đoạn khóa luận, kiến trúc được mở rộng thành nhiều Agent, mỗi Agent đảm nhiệm một
nhóm kỹ năng riêng, và khâu rà soát của người dùng được loại bỏ.

Để bước chuyển này là mở rộng thay vì hiện thực lại, kiến trúc tại tuần 8 phải bảo đảm ba ràng buộc:

1. **Sổ đăng ký công cụ tách rời khỏi bộ điều phối.** Việc bổ sung Agent chỉ yêu cầu thêm bộ điều
   phối sử dụng chung sổ công cụ, không yêu cầu sửa đổi các công cụ hiện có.
2. **Trạng thái được lưu trong cơ sở dữ liệu.** Nhiều Agent phải cùng truy cập được một trạng thái
   xử lý thống nhất.
3. **Mọi quyết định được ghi kèm căn cứ và độ tin cậy.** Cơ chế kiểm chứng chéo giữa các Agent ở
   giai đoạn khóa luận sử dụng chính dữ liệu này để phát hiện bất đồng.

Ngoài ra, toàn bộ các lần người dùng ghi đè kết quả trong giai đoạn Dự án CNTT phải được lưu trữ.
Tập dữ liệu này là cơ sở đánh giá mức độ chính xác của hệ thống tự động hoàn toàn ở giai đoạn
khóa luận.
