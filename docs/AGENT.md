# KIẾN TRÚC AGENT KIỂM CHỨNG

**Đề tài:** Xây dựng hệ thống đối soát hóa đơn và chứng từ nhập xuất (ĐA161)

Tài liệu này mô tả kiến trúc thành phần Agent trong hệ thống. Phần hiện thực được thực hiện ở
tuần 9 theo kế hoạch.

Agent không thay thế mô hình đồ thị và không tham gia vào quá trình trích xuất ban đầu. Agent
tiếp nhận kết quả đầu ra của mô hình và thực hiện kiểm chứng, hiệu chỉnh ở mức toàn tài liệu.

---

## 1. Cơ sở đề xuất

Mô hình đồ thị phân loại từng nút dựa trên đặc trưng của nút đó và các nút lân cận. Cách tiếp
cận này không có cơ chế bảo đảm tính nhất quán ở mức toàn tài liệu, dẫn đến một nhóm sai sót
có thể dự đoán trước:

| Sai sót | Nguyên nhân | Xử lý của Agent |
|---|---|---|
| Hai nút cùng được gán một loại trường vốn chỉ xuất hiện một lần | Mô hình quyết định cục bộ, không so sánh toàn cục | Chọn phương án đúng dựa trên ngữ cảnh và vị trí |
| Tổng tiền không bằng tổng các dòng | Mô hình không kiểm tra quan hệ số học giữa các trường | Phát hiện mâu thuẫn, đọc lại vùng liên quan |
| Thiếu trường bắt buộc | Không nút nào đạt ngưỡng tin cậy | Tìm bổ sung bằng suy luận trên văn bản gốc |
| Giá trị hợp lệ về hình thức nhưng bất khả thi | Mô hình không có tri thức nghiệp vụ | Từ chối và yêu cầu trích xuất lại |
| Dòng hàng bị gom sai | Quan hệ giữa các trường vượt phạm vi lân cận của nút | Gom lại theo bố cục và ngữ nghĩa |

Đây là nhóm sai sót cần suy luận ở mức tài liệu, nằm ngoài khả năng của mô hình phân loại nút.

## 2. Vị trí trong luồng xử lý

```
Tài liệu ──► OCR ──► Đồ thị ──► Mô hình đồ thị ──► Agent kiểm chứng ──► Kết quả
                                       │                   │
                                   nhãn từng nút     kiểm tra nhất quán
                                   kèm độ tin cậy    hiệu chỉnh, bổ sung
                                                     ghi vết can thiệp
```

Agent chỉ được kích hoạt khi có dấu hiệu bất thường, không chạy trên mọi tài liệu. Điều kiện
kích hoạt gồm: tồn tại trường trùng lặp, vi phạm ràng buộc số học, thiếu trường bắt buộc, hoặc
độ tin cậy trung bình dưới ngưỡng.

## 3. Bộ công cụ

Agent chỉ được gọi các công cụ đã đăng ký, với chữ ký đầu vào và đầu ra xác định. Agent không
truy vấn trực tiếp cơ sở dữ liệu và không sửa trực tiếp kết quả của mô hình.

| Công cụ | Đầu vào | Đầu ra |
|---|---|---|
| `get_predictions` | Mã tài liệu | Danh sách trường đã trích xuất kèm độ tin cậy |
| `get_page_text` | Mã tài liệu, số trang | Toàn bộ từ và tọa độ trên trang |
| `check_arithmetic` | Tập trường tiền tệ | Danh sách vi phạm ràng buộc số học |
| `find_candidates` | Loại trường cần tìm | Các nút có khả năng thuộc loại trường đó |
| `revise_field` | Mã trường, giá trị mới, căn cứ | Ghi nhận hiệu chỉnh kèm nhật ký |
| `reject_field` | Mã trường, căn cứ | Loại bỏ trường không hợp lệ |
| `regroup_line_items` | Danh sách trường dòng hàng | Phương án gom nhóm mới |

## 4. Nguyên tắc can thiệp

1. Agent chỉ hiệu chỉnh khi có **căn cứ kiểm chứng được**, không hiệu chỉnh dựa trên phỏng đoán.
2. Mọi hiệu chỉnh đều ghi lại giá trị trước và sau, kèm lý do.
3. Kết quả gốc của mô hình **không bị xóa**, cho phép đối chiếu và đo lường tác động.
4. Số lần can thiệp trên một tài liệu có giới hạn để tránh lặp vô hạn.
5. Khi không đủ căn cứ, Agent giữ nguyên kết quả của mô hình thay vì đoán.

## 5. Cơ chế ghi vết

Mỗi lượt gọi công cụ ghi một bản ghi vào bảng `audit_event`, gồm: tên công cụ, dữ liệu đầu vào,
dữ liệu đầu ra, độ tin cậy và căn cứ can thiệp.

Nhật ký này phục vụ hai mục đích: bảo đảm mọi kết quả đều truy vết được nguyên nhân, và cung cấp
dữ liệu để phân tích các trường hợp Agent hiệu chỉnh đúng cũng như hiệu chỉnh sai.

## 6. Phương pháp đánh giá

Đóng góp của Agent được đo bằng cách so sánh trên cùng một tập kiểm định:

| Cấu hình | Chỉ số |
|---|---|
| Chỉ mô hình đồ thị | AP, F1, macro-F1 |
| Mô hình đồ thị kết hợp Agent | AP, F1, macro-F1 |

Ngoài chênh lệch tổng thể, cần báo cáo thêm ba số liệu: tỷ lệ tài liệu được Agent can thiệp,
tỷ lệ can thiệp làm kết quả tốt lên, và tỷ lệ can thiệp làm kết quả xấu đi. Chỉ số thứ ba là
quan trọng nhất, vì một lớp kiểm chứng làm hỏng kết quả đúng thì không có giá trị sử dụng.

## 7. Ràng buộc thiết kế hướng tới giai đoạn khóa luận

Ở giai đoạn khóa luận, kiến trúc được mở rộng thành nhiều Agent, mỗi Agent đảm nhiệm một nhóm
kỹ năng riêng, và cơ chế kiểm chứng chéo giữa các Agent thay thế cho khâu rà soát của người dùng.

Để bước chuyển này là mở rộng thay vì hiện thực lại, kiến trúc phải bảo đảm ba ràng buộc:

1. **Sổ đăng ký công cụ tách rời khỏi bộ điều phối.** Việc bổ sung Agent chỉ yêu cầu thêm bộ
   điều phối sử dụng chung sổ công cụ.
2. **Trạng thái xử lý lưu trong cơ sở dữ liệu.** Nhiều Agent phải cùng truy cập được một trạng
   thái thống nhất.
3. **Mọi quyết định ghi kèm căn cứ và độ tin cậy.** Cơ chế kiểm chứng chéo sử dụng chính dữ liệu
   này để phát hiện bất đồng giữa các Agent.
