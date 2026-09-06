# ĐẶC TẢ HỆ THỐNG

**Đề tài:** Xây dựng hệ thống đối soát hóa đơn và chứng từ nhập xuất (ĐA161)

Tài liệu này đặc tả phạm vi chức năng, phương pháp và các quy tắc nghiệp vụ của hệ thống,
theo định hướng đã thống nhất với giảng viên hướng dẫn tại buổi họp ngày 29/08/2026.

---

## 1. Bài toán

Hệ thống thực hiện **trích xuất thông tin có cấu trúc từ tài liệu nghiệp vụ** bằng phương pháp
biểu diễn tài liệu dưới dạng đồ thị và học máy trên đồ thị. Kết quả trích xuất được sử dụng cho
lớp ứng dụng **đối soát chứng từ**.

Hai bài toán con theo chuẩn của bộ dữ liệu DocILE:

| Bài toán | Nội dung |
|---|---|
| **KILE** | Xác định vị trí và phân loại các trường thông tin trên tài liệu |
| **LIR** | Gom các trường thông tin thành từng dòng hàng |

Lớp ứng dụng đối soát sử dụng các trường đã trích xuất để đối chiếu hóa đơn với chứng từ nhập
xuất và đơn đặt hàng, phát hiện dữ liệu không nhất quán, sai lệch hoặc bất thường.

## 2. Dữ liệu

Bộ dữ liệu **DocILE**, phần `annotated-trainval`.

| Chỉ tiêu | Giá trị |
|---|---|
| Tổng số tài liệu | 5.680 (huấn luyện 5.180, kiểm định 500) |
| Số trang mỗi tài liệu | 1 trang 77%, 2 trang 19%, 3 trang 4% |
| Số từ mỗi trang | trung bình 225, lớn nhất 1.906 |
| Số trường KILE mỗi tài liệu | trung bình 12,7 |
| Số dòng hàng mỗi tài liệu | trung bình 6,8 |
| Số loại trường | KILE 36 loại, LIR 18 loại |

Mỗi tài liệu gồm ba thành phần: tệp PDF gốc, kết quả OCR có sẵn kèm tọa độ từng từ, và phần
chú giải xác định vị trí cùng loại của từng trường thông tin.

### 2.1. Kết quả khảo sát dữ liệu

Ba kết quả khảo sát có ảnh hưởng trực tiếp đến thiết kế:

**Mức độ khớp giữa trường chú giải và từ OCR: 95,64%.** Trong 24.223 trường được khảo sát,
23.168 trường khớp được với ít nhất một từ do OCR trả về; 1.055 trường không khớp được từ nào.
Đây là **trần lý thuyết** của phương pháp phân loại nút trên đồ thị từ: không mô hình nào thuộc
nhóm này vượt được ngưỡng đó, vì phần còn lại không có nút nào biểu diễn được.

**26% số trường trải trên nhiều từ.** Phân bố số từ tạo nên một trường: 1 từ chiếm 73,7%,
2 từ 7,1%, 3 từ 6,0%, 4 từ 3,0%, 5 từ 2,0%, từ 6 từ trở lên 8,2%. Do đó nhãn của nút phải theo
lược đồ BIO để khôi phục được ranh giới trường, thay vì phân loại từng nút độc lập.

**Phân bố nhãn mất cân bằng nghiêm trọng.** Mười loại trường phổ biến nhất chiếm khoảng 79%
tổng số trường, trong khi nhiều loại chỉ xuất hiện dưới 0,1%. Quá trình huấn luyện phải áp dụng
trọng số theo lớp; kết quả phải báo cáo cả macro-F1 lẫn micro-F1.

## 3. Biểu diễn đồ thị

| Thành phần | Định nghĩa |
|---|---|
| Nút | Một từ do OCR trả về |
| Đặc trưng nút | Biểu diễn văn bản, tọa độ chuẩn hóa, kích thước hộp bao, độ tin cậy OCR |
| Cạnh | Quan hệ không gian giữa các từ: k láng giềng gần nhất, cùng dòng, trên và dưới |
| Nhãn nút | Loại trường theo lược đồ BIO, suy ra bằng cách đối chiếu hộp bao của từ với chú giải |

Ở giai đoạn mở rộng, đặc trưng nút được bổ sung **đặc trưng thị giác trích bằng mạng tích chập**
từ vùng ảnh tương ứng với từ đó.

## 4. Mô hình

**Bài toán chính:** phân loại nút bằng mạng tích chập đồ thị, đầu ra là nhãn BIO cho từng từ.
Các từ liền kề cùng nhãn được gộp lại thành một trường hoàn chỉnh.

**Bài toán mở rộng:** dự đoán cạnh giữa các trường để gom thành dòng hàng, phục vụ LIR.

Tiến trình thực hiện chia làm ba bước, mỗi bước là một kết quả đo được độc lập:

1. Đồ thị với đặc trưng văn bản và tọa độ
2. Bổ sung đặc trưng thị giác từ mạng tích chập
3. Bổ sung dự đoán cạnh cho bài toán gom dòng hàng

## 5. Lớp Agent kiểm chứng

Mô hình đồ thị phân loại từng nút một cách cục bộ và không có khái niệm về tính nhất quán ở mức
toàn tài liệu. Lớp Agent xử lý đúng nhóm sai sót này.

| Sai sót của mô hình | Xử lý của Agent |
|---|---|
| Hai nút cùng được gán một loại trường chỉ xuất hiện một lần | Chọn phương án đúng dựa trên ngữ cảnh |
| Tổng tiền không khớp tổng các dòng | Phát hiện mâu thuẫn, đọc lại và hiệu chỉnh |
| Thiếu trường bắt buộc, không nút nào đạt ngưỡng | Tìm bổ sung bằng suy luận trên văn bản gốc |
| Giá trị hợp lệ về hình thức nhưng bất khả thi về nghiệp vụ | Từ chối và trích xuất lại |
| Dòng hàng bị gom sai | Gom lại theo bố cục và ngữ nghĩa |

Mọi can thiệp của Agent được ghi vào bảng nhật ký kèm căn cứ. Kiến trúc chi tiết trình bày trong
tài liệu [AGENT.md](AGENT.md).

## 6. Lớp ứng dụng đối soát

Các trường đã trích xuất được đưa vào lớp đối soát, đối chiếu hóa đơn với chứng từ nhập xuất và
đơn đặt hàng trên bộ dữ liệu tổng hợp do hệ thống tự sinh.

Năm nhóm kết quả đối soát:

| Mã | Nhãn | Điều kiện |
|---|---|---|
| `MATCHED` | Trùng khớp | Mọi chênh lệch nằm trong ngưỡng dung sai |
| `PARTIAL` | Trùng khớp một phần | Chênh lệch theo hướng thiếu |
| `MISMATCH` | Sai hoàn toàn | Chênh lệch theo hướng thừa hoặc vi phạm ràng buộc số học |
| `MISSING` | Thiếu chứng từ | Không xác định được chứng từ đối ứng |
| `DUPLICATE` | Lặp chứng từ | Trùng theo mã băm tệp hoặc khóa nghiệp vụ |

Kết quả đối soát được tính từ chênh lệch định lượng và độ phủ, không tính từ điểm tương đồng.

## 7. Phương pháp đánh giá

**Trích xuất.** Đánh giá trên tập kiểm định theo đúng chuẩn của DocILE: AP, F1, độ chính xác và
độ bao phủ, tính trung bình trên toàn bộ trường và loại trường. Báo cáo song song macro-F1 để
phản ánh các loại trường hiếm.

**Đóng góp của từng thành phần.** Ba cấu hình được đo trên cùng một tập kiểm định:

| Cấu hình | Mục đích đo |
|---|---|
| Đồ thị với đặc trưng văn bản và tọa độ | Kết quả cơ sở |
| Bổ sung đặc trưng thị giác | Mức đóng góp của thành phần thị giác |
| Bổ sung lớp Agent kiểm chứng | Mức đóng góp của thành phần suy luận |

**So sánh với công trình đã công bố.** Đối chiếu với các mô hình nền công bố kèm bộ dữ liệu.

## 8. Quy ước kỹ thuật bắt buộc

1. Giá trị tiền tệ và số lượng sử dụng kiểu `Decimal` trong mã nguồn và `NUMERIC` trong cơ sở
   dữ liệu. Không sử dụng kiểu dấu chấm động.
2. Tọa độ vùng văn bản được lưu cùng mọi giá trị trích xuất.
3. Cấu hình huấn luyện, phiên bản mô hình và hạt giống ngẫu nhiên được ghi lại đầy đủ để bảo
   đảm kết quả tái lập được.
4. Bộ dữ liệu đặt ngoài kho mã nguồn, truy cập qua biến môi trường `DOCILE_DIR`.
5. Độ tin cậy chỉ được dùng để định tuyến sang bước kiểm chứng, không dùng để gán kết quả.
