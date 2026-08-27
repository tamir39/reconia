# SPEC — Reconia (ĐA161)

Đặc tả rút gọn. Bám sát mô tả đề tài đã đăng ký trên ELIT và định hướng của GVHD.

---

## 1. Bài toán

Đối soát **hóa đơn** với **chứng từ nhập xuất** (phiếu nhập kho, phiếu xuất kho) và
**đơn đặt hàng**, phát hiện dữ liệu không nhất quán, sai lệch hoặc bất thường.

Đối soát ba chiều:

| Chứng từ | Trả lời câu hỏi |
|---|---|
| Đơn đặt hàng | Đã **đặt** bao nhiêu? |
| Phiếu nhập/xuất kho | Thực tế đã **nhận / xuất** bao nhiêu? |
| Hóa đơn | Bị **tính tiền** bao nhiêu? |

Chênh lệch giữa ba con số này là nơi phát sinh thất thoát.

## 2. Sáu module

Giữ nguyên pipeline theo định hướng của GVHD. Mỗi module tương ứng một thư mục mã nguồn
và một phạm vi (scope) trong quy ước commit.

| # | Module | Thư mục | Nhiệm vụ | Tuần |
|---|---|---|---|---|
| 1 | Tiếp nhận, phân loại và tiền xử lý | `app/intake/` | Nhận PDF/ảnh/XML, kiểm tra chất lượng, nhận diện loại chứng từ, xoay và khử nhiễu, phát hiện tệp trùng | 2 |
| 2 | Trích xuất theo lược đồ và chuẩn hóa | `app/extraction/` | Trích xuất theo lược đồ dữ liệu, lưu bằng chứng từng trường, chuẩn hóa ngày/tiền/thuế/ĐVT/tên sản phẩm | 3 |
| 3 | Đối soát và ghép nối | `app/matching/` | Ghép Hóa đơn – Phiếu kho – Đơn đặt hàng qua ba tầng, trả về điểm tin cậy và lý do | 4 |
| 4 | Kiểm tra ràng buộc và phát hiện bất thường | `app/rules/` | 12 quy tắc ràng buộc, phân loại 5 nhóm kết quả, phát hiện trùng lặp và biến động giá | 5 |
| 5 | Agent điều phối và rà soát của con người | `app/agent/` | Điều phối pipeline theo trạng thái, định tuyến theo độ tin cậy, tạm dừng chờ người duyệt | 6, 8 |
| 6 | Nhật ký, bộ nhớ hiệu chỉnh và giám sát | `app/audit/` | Ghi vết mọi quyết định, truy hồi các hiệu chỉnh đã xác nhận | 11 |

Chi tiết kiến trúc Module 5 xem [AGENT.md](AGENT.md).

## 3. Năm kết quả đối soát

Cố định theo mô tả đề tài, không được đổi tên.

| Mã | Nhãn | Khi nào |
|---|---|---|
| `MATCHED` | Trùng khớp | Mọi chênh lệch nằm trong dung sai |
| `PARTIAL` | Trùng khớp một phần | Lệch **thiếu**: giao thiếu, xuất hóa đơn thiếu — luồng nghiệp vụ bình thường |
| `MISMATCH` | Sai hoàn toàn | Lệch **thừa**: xuất hóa đơn vượt, sai đơn giá, sai số học — là mâu thuẫn |
| `MISSING` | Thiếu chứng từ | Không tìm được chứng từ đối ứng |
| `DUPLICATE` | Lặp chứng từ | Trùng theo mã băm tệp, khóa nghiệp vụ, hoặc nghi ngờ trùng |

> **Quy tắc bắt buộc:** kết quả tính từ **chênh lệch và độ phủ**, không tính từ điểm tương đồng.
> Điểm tương đồng chỉ quyết định hai dòng có **so sánh được với nhau hay không**;
> chênh lệch mới quyết định kết luận.
> Nếu ánh xạ ngưỡng tương đồng thành kết luận thì một tên hàng lạ sẽ bị gắn *sai hoàn toàn*
> trên một hóa đơn hoàn toàn đúng.

> **Chiều lệch quyết định PARTIAL hay MISMATCH.** Sai chỗ này thì mọi lần giao thiếu
> đều bị báo như gian lận.

## 4. Bốn đường trích xuất

Xếp theo mức độ tin cậy. **Không bao giờ chạy OCR với chứng từ đã có XML hoặc đã có lớp văn bản.**

| # | Đường | Công cụ | Độ tin cậy |
|---|---|---|---|
| 1 | Hóa đơn điện tử XML | lxml | 1.00 — chính xác tuyệt đối |
| 2 | PDF có lớp văn bản | pdfplumber | Chính xác từng ký tự, kèm tọa độ |
| 3 | Ảnh / PDF scan | PaddleOCR + VietOCR (cục bộ, CPU) | Có sai số |
| 4 | Nhập tay | Con người | 1.00, có ghi nhật ký |

Hóa đơn điện tử theo Nghị định 123/2020 và Thông tư 78/2021 là **XML** — bản PDF chỉ là
*bản thể hiện*. Phiếu nhập/xuất kho theo Thông tư 200/2014 hoặc 133/2016 **không có XML**,
nên đó mới là chỗ thật sự cần OCR.

> Các văn bản pháp lý trên phải **kiểm tra lại hiệu lực** trước khi trích dẫn vào báo cáo.

## 5. Ghép nối ba tầng

| Tầng | Cách làm | Độ tin cậy |
|---|---|---|
| 0 | Đọc tham chiếu in sẵn trên phiếu kho: *"theo hóa đơn số ... ngày ..."* | 1.00, không cần khớp gần đúng |
| 1 | Chặn ứng viên theo mã số thuế và khoảng lệch ngày | — |
| 2 | Tổng hợp theo ô `(mã sản phẩm, đơn vị cơ sở)` rồi ghép dòng tham lam | Có điểm số |

Tầng 2 làm **bài toán N:M tự tan**: một hóa đơn ứng với ba phiếu nhập trở thành phép so sánh
giữa hai con số tổng hợp, không cần thuật toán gán tối ưu.

**Đo và ghi vào báo cáo tỷ lệ cặp được giải quyết ngay ở tầng 0** — hầu hết nhóm bỏ qua
tầng này rồi mất nhiều tuần cho khớp gần đúng.

## 6. Định danh

- **Đối tác: chỉ dùng mã số thuế.** Không bao giờ so khớp theo địa chỉ — Việt Nam đổi sang
  mô hình hành chính hai cấp từ 01/07/2025, nên cùng một nhà cung cấp có hai địa chỉ khác
  nhau và **cả hai đều đúng**.
- **Sản phẩm**, dừng ở bước đầu tiên khớp được:
  1. Mã hàng của nhà cung cấp (có sẵn trong XML — khóa khớp giá trị nhất, ít nhóm để ý)
  2. Tên đã chuẩn hóa (chuẩn hóa Unicode NFC trước)
  3. Bảng bí danh `(đối tác, tên đã chuẩn hóa)`
  4. Cổng thuộc tính: dung tích và quy cách đóng gói **phải khớp tuyệt đối**
  5. Khớp gần đúng RapidFuzz
  6. Con người xác nhận → **ghi ngược vào bảng bí danh**

> Cổng thuộc tính là bắt buộc: *"Bia Sài Gòn 330ml"* và *"Bia Sài Gòn 500ml"* gần như
> giống hệt nhau về mặt chữ nhưng là hai mặt hàng khác nhau. **Con số quyết định, không phải ngữ nghĩa.**

## 7. Phát hiện trùng lặp — ba lớp

| Lớp | Cách phát hiện |
|---|---|
| L0 | Mã băm SHA-256 của tệp — cùng một tệp tải lên hai lần |
| L1 | Khóa nghiệp vụ `(mã số thuế, ký hiệu, số hóa đơn)`, ràng buộc UNIQUE ở CSDL |
| L2 | Cùng đối tác, cùng ngày, tổng tiền trong dung sai, khác số hóa đơn ⇒ **nghi ngờ trùng** |

L2 chính là ca *"cùng một hóa đơn scan hai lần, sai một chữ số"*.

## 8. Mười hai quy tắc kiểm tra

Mỗi quy tắc là **một hàm thuần** trong `app/rules/`, ghi kết quả vào bảng `finding`
với `rule_code` tương ứng. Cột *Kết quả* cho biết quy tắc đó góp phần tạo ra nhóm nào.

| # | `rule_code` | Kiểm tra | Mức | Kết quả |
|---|---|---|---|---|
| 1 | `LINE_ARITHMETIC` | Số lượng × đơn giá = thành tiền | ERROR | MISMATCH |
| 2 | `SUBTOTAL_SUM` | Tổng tiền hàng = Σ thành tiền các dòng | ERROR | MISMATCH |
| 3 | `TAX_AMOUNT` | Tiền thuế = tiền hàng × thuế suất | ERROR | MISMATCH |
| 4 | `TOTAL_SUM` | Tổng thanh toán = tiền hàng + tiền thuế | ERROR | MISMATCH |
| 5 | `TAX_RATE_VALID` | Thuế suất thuộc {0%, 5%, 8%, 10%} | WARNING | — |
| 6 | `TAX_CODE_FORMAT` | Mã số thuế đúng cấu trúc 10 hoặc 13 chữ số | WARNING | — |
| 7 | `QTY_VARIANCE` | Chênh lệch số lượng giữa các chứng từ vượt dung sai | ERROR | PARTIAL / MISMATCH |
| 8 | `PRICE_VARIANCE` | Chênh lệch đơn giá vượt dung sai | ERROR | MISMATCH |
| 9 | `QTY_OVER_ORDER` | Số lượng nhận hoặc xuất hóa đơn vượt số lượng đã đặt | ERROR | MISMATCH |
| 10 | `MISSING_COUNTERPART` | Thiếu chứng từ đối ứng trong kỳ | ERROR | MISSING |
| 11 | `DUPLICATE_DOC` | Trùng chứng từ theo ba lớp ở mục 7 | ERROR | DUPLICATE |
| 12 | `PERIOD_MISMATCH` | Ngày hóa đơn và ngày phiếu kho lệch kỳ kế toán | WARNING | — |

> Quy tắc 1 vừa là kiểm tra nghiệp vụ vừa là **phép tự kiểm bộ phân tích số**: lệch khoảng
> 1000 lần nghĩa là đã đọc nhầm `1.234.567,89` thành `1,234,567.89`.

Phát hiện bất thường bằng thống kê (biến động đơn giá, số lượng) nằm ngoài 12 quy tắc này,
dùng độ lệch tuyệt đối trung vị trên các giao dịch gần nhất của cùng cặp `(đối tác, sản phẩm)`.

## 9. Dung sai

Hai mức, chỉnh được từ giao diện: mức **cảnh báo** và mức **vi phạm**.
Áp dụng cho số lượng, đơn giá và tổng tiền. Không có dung sai thì làm tròn sẽ tạo ra
hàng loạt sai lệch giả.

## 10. Quy ước kỹ thuật bắt buộc

- Tiền và số lượng dùng `Decimal` / `NUMERIC`. **Không bao giờ dùng `float`.**
- Lưu tọa độ vùng chữ ngay từ lần trích xuất đầu tiên.
- Độ tin cậy được **suy ra**, không tự khai báo: bộ trích xuất bằng luật và bộ bằng mô hình
  có đồng ý nhau không, phép số học có đúng không, bộ nhận dạng trả về bao nhiêu.
- Độ tin cậy chỉ dùng để **đưa vào hàng đợi rà soát**. Nó **không bao giờ** gán kết quả đối soát.
