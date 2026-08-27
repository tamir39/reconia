# SPEC — Reconia (ĐA161)

Đặc tả rút gọn. Bám sát mô tả đề tài đã đăng ký trên ELIT.

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

## 2. Năm kết quả đối soát

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

## 3. Bốn đường trích xuất

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

## 4. Ghép nối ba tầng

| Tầng | Cách làm | Độ tin cậy |
|---|---|---|
| 0 | Đọc tham chiếu in sẵn trên phiếu kho: *"theo hóa đơn số ... ngày ..."* | 1.00, không cần khớp gần đúng |
| 1 | Chặn ứng viên theo mã số thuế và khoảng lệch ngày | — |
| 2 | Tổng hợp theo ô `(mã sản phẩm, đơn vị cơ sở)` rồi ghép dòng tham lam | Có điểm số |

Tầng 2 làm **bài toán N:M tự tan**: một hóa đơn ứng với ba phiếu nhập trở thành phép so sánh
giữa hai con số tổng hợp, không cần thuật toán gán tối ưu.

**Đo và ghi vào báo cáo tỷ lệ cặp được giải quyết ngay ở tầng 0** — hầu hết nhóm bỏ qua
tầng này rồi mất nhiều tuần cho khớp gần đúng.

## 5. Định danh

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

## 6. Phát hiện trùng lặp — ba lớp

| Lớp | Cách phát hiện |
|---|---|
| L0 | Mã băm SHA-256 của tệp — cùng một tệp tải lên hai lần |
| L1 | Khóa nghiệp vụ `(mã số thuế, ký hiệu, số hóa đơn)`, ràng buộc UNIQUE ở CSDL |
| L2 | Cùng đối tác, cùng ngày, tổng tiền trong dung sai, khác số hóa đơn ⇒ **nghi ngờ trùng** |

L2 chính là ca *"cùng một hóa đơn scan hai lần, sai một chữ số"*.

## 7. Dung sai

Hai mức, chỉnh được từ giao diện: mức **cảnh báo** và mức **vi phạm**.
Áp dụng cho số lượng, đơn giá và tổng tiền. Không có dung sai thì làm tròn sẽ tạo ra
hàng loạt sai lệch giả.

## 8. Quy ước kỹ thuật bắt buộc

- Tiền và số lượng dùng `Decimal` / `NUMERIC`. **Không bao giờ dùng `float`.**
- Lưu tọa độ vùng chữ ngay từ lần trích xuất đầu tiên.
- Kiểm tra `số lượng × đơn giá = thành tiền` vừa là quy tắc nghiệp vụ vừa là **phép tự kiểm
  bộ phân tích số**: lệch khoảng 1000 lần nghĩa là đọc nhầm `1.234.567,89` thành `1,234,567.89`.
- Độ tin cậy được **suy ra**, không tự khai báo: bộ trích xuất bằng luật và bộ bằng mô hình
  có đồng ý nhau không, phép số học có đúng không, bộ nhận dạng trả về bao nhiêu.
- Độ tin cậy chỉ dùng để **đưa vào hàng đợi rà soát**. Nó **không bao giờ** gán kết quả đối soát.
