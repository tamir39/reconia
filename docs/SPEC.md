# ĐẶC TẢ HỆ THỐNG

**Đề tài:** Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin
từ hóa đơn (ĐA161)

*Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction*

Tài liệu này đặc tả phạm vi và phương pháp thực hiện, theo tên và mô tả đề tài đã được giảng
viên hướng dẫn cập nhật trên hệ thống ELIT.

> **Trạng thái:** đề tài chưa được duyệt. Một số nội dung còn chờ xác nhận của giảng viên
> hướng dẫn, đã ghi rõ tại từng mục.

---

## 1. Bài toán

Đề tài thực hiện **đánh giá và so sánh các phương pháp học sâu** cho bài toán trích xuất thông
tin từ hóa đơn, theo yêu cầu tại mô tả đề tài:

1. Khảo sát và lựa chọn từ ba đến năm mô hình học sâu.
2. Thực nghiệm trên một số bộ dữ liệu hóa đơn.
3. Cài đặt hoặc chạy lại các mô hình dưới **điều kiện thực nghiệm thống nhất**.
4. Đánh giá bằng Precision, Recall và F1-score.
5. So sánh và phân tích ưu điểm, hạn chế của từng phương pháp.

Bài toán trích xuất được xét theo chuẩn của bộ dữ liệu DocILE:

| Bài toán | Nội dung |
|---|---|
| **KILE** | Xác định vị trí và phân loại các trường thông tin trên tài liệu |
| **LIR** | Gom các trường thông tin thành từng dòng hàng |

**Đóng góp chính của đề tài là khung thực nghiệm thống nhất.** Kết quả công bố trong các công
trình khác nhau thường không so sánh trực tiếp được với nhau do khác biệt về cách chia tập dữ
liệu, cách tiền xử lý và cách tính chỉ số. Đề tài đưa các mô hình về cùng một điều kiện đo.

### 1.1. Các mô hình dự kiến so sánh

| Nhóm phương pháp | Mô hình | Nơi huấn luyện |
|---|---|---|
| Đồ thị | Mạng tích chập đồ thị do nhóm hiện thực | Máy cá nhân |
| Ngôn ngữ | RoBERTa kết hợp thông tin bố cục | Kaggle |
| Đa phương thức | LayoutLMv3 | Kaggle |
| Đa phương thức nhẹ | LiLT | Kaggle |

*Danh sách chờ giảng viên hướng dẫn xác nhận về số lượng và thành phần mô hình.*

### 1.2. Môi trường huấn luyện

Các mô hình Transformer được huấn luyện trên **Kaggle**, nơi cung cấp GPU có dung lượng bộ nhớ
đủ lớn. Máy cá nhân chỉ dùng cho mô hình đồ thị và cho khung đánh giá.

Cách bố trí này đặt ra ba yêu cầu kỹ thuật:

1. Bộ dữ liệu phải được tải lên Kaggle dưới dạng dataset dùng chung cho mọi lần chạy.
2. Quy trình huấn luyện phải **lưu điểm kiểm tra định kỳ**, do mỗi phiên làm việc trên Kaggle có
   giới hạn thời gian.
3. Khung đánh giá phải chạy được ở cả hai nơi, trên cùng một định dạng kết quả dự đoán, để kết
   quả của các mô hình so sánh được với nhau.

Phần cứng của từng lần chạy được ghi lại trong kết quả thực nghiệm, vì thời gian huấn luyện và
bộ nhớ yêu cầu là một trong các tiêu chí so sánh.

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

## 6. Lớp ứng dụng minh họa

Phần đối soát chứng từ **không còn nằm trong phạm vi chính** của đề tài sau khi tên đề tài được
cập nhật. Phần này được giữ lại ở mức ứng dụng minh họa, sử dụng kết quả trích xuất để đối chiếu
chứng từ trên bộ dữ liệu tổng hợp, cho thấy kết quả trích xuất dùng được trong một quy trình
nghiệp vụ thực tế.

Năm nhóm kết quả đối soát:

| Mã | Nhãn | Điều kiện |
|---|---|---|
| `MATCHED` | Trùng khớp | Mọi chênh lệch nằm trong ngưỡng dung sai |
| `PARTIAL` | Trùng khớp một phần | Chênh lệch theo hướng thiếu |
| `MISMATCH` | Sai hoàn toàn | Chênh lệch theo hướng thừa hoặc vi phạm ràng buộc số học |
| `MISSING` | Thiếu chứng từ | Không xác định được chứng từ đối ứng |
| `DUPLICATE` | Lặp chứng từ | Trùng theo mã băm tệp hoặc khóa nghiệp vụ |

Kết quả đối soát được tính từ chênh lệch định lượng và độ phủ, không tính từ điểm tương đồng.

## 7. Khung thực nghiệm thống nhất

Đây là nội dung trọng tâm của đề tài. Mọi mô hình được đo dưới cùng một điều kiện:

| Yếu tố | Quy định thống nhất |
|---|---|
| Dữ liệu | Cùng bộ dữ liệu, cùng định dạng trung gian |
| Cách chia tập | Cùng tập huấn luyện và tập kiểm định |
| Chỉ số | Precision, Recall, F1 tính theo cùng một cách |
| Tiền xử lý | Cùng kết quả OCR đầu vào |
| Phần cứng | Ghi rõ phần cứng của từng lần chạy |

Khung đánh giá được thiết kế **độc lập với kiến trúc mô hình**: bổ sung một mô hình mới chỉ yêu
cầu cung cấp đầu ra theo định dạng chung, không sửa phần đánh giá.

Nội dung báo cáo kết quả:

1. Bảng so sánh các mô hình trên từng bộ dữ liệu.
2. Kết quả theo từng loại trường, kèm macro-F1 để phản ánh các loại trường hiếm.
3. Chi phí thực nghiệm: thời gian huấn luyện và bộ nhớ GPU yêu cầu.
4. Phân tích ưu điểm và hạn chế của từng phương pháp.
5. Phân tích các trường hợp lỗi điển hình.

## 8. Quy ước kỹ thuật bắt buộc

1. Giá trị tiền tệ và số lượng sử dụng kiểu `Decimal` trong mã nguồn và `NUMERIC` trong cơ sở
   dữ liệu. Không sử dụng kiểu dấu chấm động.
2. Tọa độ vùng văn bản được lưu cùng mọi giá trị trích xuất.
3. Cấu hình huấn luyện, phiên bản mô hình và hạt giống ngẫu nhiên được ghi lại đầy đủ để bảo
   đảm kết quả tái lập được.
4. Bộ dữ liệu đặt ngoài kho mã nguồn, truy cập qua biến môi trường `DOCILE_DIR`.
5. Độ tin cậy chỉ được dùng để định tuyến sang bước kiểm chứng, không dùng để gán kết quả.
