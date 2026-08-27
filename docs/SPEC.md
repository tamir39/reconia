# ĐẶC TẢ HỆ THỐNG

**Đề tài:** Xây dựng hệ thống đối soát hóa đơn và chứng từ nhập xuất (ĐA161)

Tài liệu này đặc tả phạm vi chức năng, cấu trúc pipeline và các quy tắc nghiệp vụ của hệ thống.
Nội dung bám sát mô tả đề tài đã đăng ký và định hướng của giảng viên hướng dẫn.

---

## 1. Bài toán

Hệ thống thực hiện đối soát **hóa đơn** với **chứng từ nhập xuất** (phiếu nhập kho, phiếu xuất
kho) và **đơn đặt hàng**, nhằm phát hiện các trường hợp dữ liệu không nhất quán, sai lệch hoặc
bất thường.

Quá trình đối soát được thực hiện trên ba nguồn chứng từ:

| Chứng từ | Đại lượng xác định |
|---|---|
| Đơn đặt hàng | Số lượng đã đặt |
| Phiếu nhập kho / phiếu xuất kho | Số lượng thực nhận hoặc thực xuất |
| Hóa đơn | Số lượng và giá trị được tính tiền |

Chênh lệch giữa ba đại lượng này là căn cứ để xác định các sai lệch trong quy trình mua hàng
và quản lý kho.

## 2. Cấu trúc sáu module

Hệ thống được tổ chức thành sáu module, mỗi module tương ứng với một thư mục mã nguồn và một
phạm vi trong quy ước quản lý phiên bản.

| # | Module | Thư mục | Chức năng | Tuần |
|---|---|---|---|---|
| 1 | Tiếp nhận, phân loại và tiền xử lý | `app/intake/` | Tiếp nhận tệp PDF, ảnh và XML; kiểm tra chất lượng; nhận diện loại chứng từ; hiệu chỉnh và khử nhiễu ảnh; phát hiện tệp trùng lặp | 2 |
| 2 | Trích xuất theo lược đồ và chuẩn hóa | `app/extraction/` | Trích xuất dữ liệu theo lược đồ; lưu bằng chứng theo từng trường; chuẩn hóa ngày tháng, số tiền, thuế suất, đơn vị tính và tên sản phẩm | 3 |
| 3 | Đối soát và ghép nối | `app/matching/` | Ghép nối hóa đơn, phiếu kho và đơn đặt hàng theo ba tầng; trả về điểm tin cậy và căn cứ ghép nối | 4 |
| 4 | Kiểm tra ràng buộc và phát hiện bất thường | `app/rules/` | Áp dụng mười hai quy tắc ràng buộc; phân loại năm nhóm kết quả; phát hiện trùng lặp và biến động giá | 5 |
| 5 | Agent điều phối và rà soát của người dùng | `app/agent/` | Điều phối pipeline theo trạng thái; định tuyến theo độ tin cậy; tạm dừng chờ người dùng phê duyệt | 6, 8 |
| 6 | Nhật ký, bộ nhớ hiệu chỉnh và giám sát | `app/audit/` | Ghi nhận toàn bộ quyết định; truy hồi các hiệu chỉnh đã được xác nhận | 11 |

Kiến trúc chi tiết của Module 5 được trình bày trong tài liệu [AGENT.md](AGENT.md).

## 3. Năm nhóm kết quả đối soát

Tập giá trị kết quả được xác định theo mô tả đề tài đã đăng ký.

| Mã | Nhãn | Điều kiện |
|---|---|---|
| `MATCHED` | Trùng khớp | Mọi chênh lệch nằm trong ngưỡng dung sai |
| `PARTIAL` | Trùng khớp một phần | Chênh lệch theo hướng thiếu: giao thiếu hoặc xuất hóa đơn thiếu so với thực nhận |
| `MISMATCH` | Sai hoàn toàn | Chênh lệch theo hướng thừa, sai lệch đơn giá, hoặc vi phạm ràng buộc số học |
| `MISSING` | Thiếu chứng từ | Không xác định được chứng từ đối ứng trong kỳ |
| `DUPLICATE` | Lặp chứng từ | Trùng lặp theo một trong ba lớp phát hiện tại mục 7 |

**Nguyên tắc phân loại.** Kết quả đối soát được tính từ **chênh lệch định lượng và độ phủ**,
không tính từ điểm tương đồng. Điểm tương đồng chỉ xác định hai dòng chứng từ có thuộc cùng
một mặt hàng để so sánh hay không; giá trị chênh lệch mới xác định kết luận. Việc ánh xạ trực
tiếp ngưỡng tương đồng thành kết luận sẽ dẫn đến phân loại sai đối với các chứng từ hợp lệ có
tên hàng khác thường.

**Nguyên tắc xác định chiều lệch.** Chênh lệch thiếu được phân loại là `PARTIAL`, chênh lệch
thừa được phân loại là `MISMATCH`. Giao hàng thiếu là tình huống nghiệp vụ thông thường, trong
khi xuất hóa đơn vượt số lượng thực nhận là mâu thuẫn cần xử lý.

## 4. Bốn đường trích xuất

Các đường trích xuất được xếp theo mức độ tin cậy giảm dần.

| # | Đường trích xuất | Công cụ | Độ tin cậy |
|---|---|---|---|
| 1 | Hóa đơn điện tử định dạng XML | lxml | Chính xác tuyệt đối |
| 2 | Tệp PDF có lớp văn bản | pdfplumber | Chính xác đến từng ký tự, kèm tọa độ |
| 3 | Ảnh hoặc PDF dạng quét | PaddleOCR kết hợp VietOCR, chạy cục bộ trên CPU | Có sai số |
| 4 | Nhập liệu thủ công | Người dùng | Chính xác, có ghi nhật ký |

**Ràng buộc bắt buộc:** hệ thống không thực hiện nhận dạng ký tự quang học đối với chứng từ đã
có tệp XML kèm theo hoặc đã có sẵn lớp văn bản.

Theo Nghị định 123/2020/NĐ-CP và Thông tư 78/2021/TT-BTC, hóa đơn điện tử có bản gốc là tệp
XML; bản PDF chỉ là bản thể hiện. Phiếu nhập kho và phiếu xuất kho theo Thông tư 200/2014/TT-BTC
hoặc Thông tư 133/2016/TT-BTC không có định dạng XML chuẩn, do đó đây là nhóm chứng từ cần đến
nhận dạng ký tự quang học.

*Hiệu lực của các văn bản pháp lý nêu trên cần được kiểm tra lại trước khi trích dẫn trong báo cáo.*

## 5. Cơ chế ghép nối ba tầng

| Tầng | Phương pháp | Độ tin cậy |
|---|---|---|
| 0 | Đọc tham chiếu tường minh in trên phiếu kho theo định dạng "theo hóa đơn số ... ngày ..." | Tuyệt đối, không sử dụng khớp gần đúng |
| 1 | Chặn tập ứng viên theo mã số thuế và khoảng lệch ngày cho phép | — |
| 2 | Tổng hợp theo cặp `(mã sản phẩm, đơn vị cơ sở)`, sau đó ghép dòng theo thứ tự tham lam | Có điểm tin cậy |

Tầng 2 xử lý trường hợp một hóa đơn tương ứng với nhiều phiếu kho bằng cách quy bài toán ghép
cặp nhiều–nhiều về phép so sánh giữa hai đại lượng tổng hợp, do đó không cần thuật toán gán tối ưu.

Tỷ lệ cặp chứng từ được giải quyết tại tầng 0 được đo và ghi nhận trong kết quả thực nghiệm.

## 6. Cơ chế định danh

**Định danh đối tác.** Hệ thống sử dụng mã số thuế làm khóa định danh duy nhất. Địa chỉ được lưu
để hiển thị nhưng không tham gia vào quá trình so khớp, do Việt Nam chuyển sang mô hình hành
chính hai cấp từ ngày 01/07/2025, dẫn đến cùng một đối tác có thể mang hai địa chỉ khác nhau và
cả hai đều hợp lệ.

**Định danh sản phẩm.** Quá trình phân giải dừng tại bước đầu tiên cho kết quả:

1. Mã hàng của nhà cung cấp, lấy từ tệp XML
2. Tên sản phẩm đã chuẩn hóa, sau khi chuẩn hóa Unicode dạng NFC
3. Bảng bí danh theo cặp `(đối tác, tên đã chuẩn hóa)`
4. Cổng thuộc tính: dung tích và quy cách đóng gói phải trùng khớp tuyệt đối
5. Khớp gần đúng bằng RapidFuzz
6. Người dùng xác nhận, kết quả được ghi ngược vào bảng bí danh

Cổng thuộc tính tại bước 4 là bắt buộc. Các tên sản phẩm chỉ khác nhau ở thông số định lượng có
mức tương đồng văn bản rất cao nhưng tương ứng với các mặt hàng khác nhau, do đó thuộc tính định
lượng giữ vai trò quyết định thay vì mức tương đồng ngữ nghĩa.

## 7. Phát hiện trùng lặp

| Lớp | Phương pháp |
|---|---|
| L0 | So sánh mã băm SHA-256 của tệp, phát hiện trường hợp tải lên cùng một tệp nhiều lần |
| L1 | Khóa nghiệp vụ `(mã số thuế, ký hiệu, số hóa đơn)` với ràng buộc duy nhất ở mức cơ sở dữ liệu |
| L2 | Cùng đối tác, cùng ngày, tổng tiền nằm trong dung sai nhưng khác số hóa đơn, xác định là nghi ngờ trùng lặp |

Lớp L2 xử lý trường hợp cùng một chứng từ được quét nhiều lần và có ký tự bị nhận dạng sai.

## 8. Mười hai quy tắc kiểm tra ràng buộc

Mỗi quy tắc được hiện thực dưới dạng một hàm thuần trong `app/rules/`, ghi kết quả vào bảng
`finding` với mã quy tắc tương ứng.

| # | Mã quy tắc | Nội dung kiểm tra | Mức | Kết quả liên quan |
|---|---|---|---|---|
| 1 | `LINE_ARITHMETIC` | Số lượng nhân đơn giá bằng thành tiền | ERROR | MISMATCH |
| 2 | `SUBTOTAL_SUM` | Tổng tiền hàng bằng tổng thành tiền các dòng | ERROR | MISMATCH |
| 3 | `TAX_AMOUNT` | Tiền thuế bằng tiền hàng nhân thuế suất | ERROR | MISMATCH |
| 4 | `TOTAL_SUM` | Tổng thanh toán bằng tiền hàng cộng tiền thuế | ERROR | MISMATCH |
| 5 | `TAX_RATE_VALID` | Thuế suất thuộc tập giá trị hợp lệ {0%, 5%, 8%, 10%} | WARNING | — |
| 6 | `TAX_CODE_FORMAT` | Mã số thuế đúng cấu trúc 10 hoặc 13 chữ số | WARNING | — |
| 7 | `QTY_VARIANCE` | Chênh lệch số lượng giữa các chứng từ vượt dung sai | ERROR | PARTIAL / MISMATCH |
| 8 | `PRICE_VARIANCE` | Chênh lệch đơn giá vượt dung sai | ERROR | MISMATCH |
| 9 | `QTY_OVER_ORDER` | Số lượng thực nhận hoặc xuất hóa đơn vượt số lượng đã đặt | ERROR | MISMATCH |
| 10 | `MISSING_COUNTERPART` | Không có chứng từ đối ứng trong kỳ | ERROR | MISSING |
| 11 | `DUPLICATE_DOC` | Trùng lặp chứng từ theo ba lớp tại mục 7 | ERROR | DUPLICATE |
| 12 | `PERIOD_MISMATCH` | Ngày hóa đơn và ngày phiếu kho thuộc hai kỳ kế toán khác nhau | WARNING | — |

Quy tắc số 1 đồng thời giữ vai trò kiểm chứng bộ phân tích số: sai lệch theo bậc nghìn cho thấy
hệ thống đã diễn giải sai quy ước dấu phân cách thập phân và dấu phân cách nghìn.

Phát hiện bất thường bằng phương pháp thống kê đối với đơn giá và số lượng nằm ngoài phạm vi mười
hai quy tắc nêu trên, sử dụng độ lệch tuyệt đối trung vị trên các giao dịch gần nhất của cùng cặp
`(đối tác, sản phẩm)`.

## 9. Dung sai

Hệ thống áp dụng hai mức dung sai có thể cấu hình từ giao diện: mức cảnh báo và mức vi phạm.
Dung sai được áp dụng cho số lượng, đơn giá và tổng tiền nhằm loại trừ các sai lệch phát sinh
do làm tròn.

## 10. Quy ước kỹ thuật bắt buộc

1. Giá trị tiền tệ và số lượng sử dụng kiểu `Decimal` trong mã nguồn và `NUMERIC` trong cơ sở
   dữ liệu. Không sử dụng kiểu dấu chấm động.
2. Tọa độ vùng văn bản được lưu ngay từ lần trích xuất đầu tiên.
3. Độ tin cậy được suy ra từ các tín hiệu kiểm chứng độc lập: mức đồng thuận giữa bộ trích xuất
   theo luật và bộ trích xuất theo mô hình, tính đúng đắn của các phép kiểm tra số học, và độ tin
   cậy do bộ nhận dạng trả về. Hệ thống không sử dụng giá trị độ tin cậy do mô hình tự khai báo.
4. Độ tin cậy chỉ được dùng để định tuyến chứng từ vào hàng đợi rà soát, không được dùng để gán
   kết quả đối soát.
