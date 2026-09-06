# TỔNG QUAN CÔNG TRÌNH LIÊN QUAN

**Đề tài:** Xây dựng hệ thống đối soát hóa đơn và chứng từ nhập xuất (ĐA161)

Tài liệu tổng hợp các công trình liên quan đến trích xuất thông tin từ tài liệu nghiệp vụ, tập
trung vào hướng biểu diễn tài liệu bằng đồ thị.

> **Lưu ý:** trước khi trích dẫn vào báo cáo, cần kiểm tra lại thông tin xuất bản của từng công
> trình từ nguồn gốc. Danh mục này phục vụ định hướng đọc, chưa phải danh mục tài liệu tham khảo
> đã hiệu đính.

---

## 1. Bối cảnh

Trích xuất thông tin từ tài liệu giàu bố cục (visually rich documents) khác với trích xuất từ
văn bản thuần túy ở chỗ ý nghĩa của một đoạn văn bản phụ thuộc vào **vị trí** của nó trên trang.
Số hóa đơn nằm ở góc trên bên phải, tổng tiền nằm cuối bảng, mã hàng nằm ở cột đầu tiên. Các
phương pháp xử lý ngôn ngữ tự nhiên theo chuỗi không tận dụng được thông tin này.

Ba hướng tiếp cận chính đã hình thành:

| Hướng | Ý tưởng | Đại diện |
|---|---|---|
| Theo chuỗi | Xếp các từ thành chuỗi rồi gán nhãn | Các mô hình gán nhãn chuỗi truyền thống |
| Theo đồ thị | Biểu diễn trang thành đồ thị, học trên cấu trúc | Liu và cộng sự (2019), PICK, Doc2Graph |
| Tiền huấn luyện đa phương thức | Huấn luyện trước trên khối lượng lớn tài liệu, kết hợp văn bản, bố cục và hình ảnh | Họ mô hình LayoutLM |

Đề tài này theo **hướng đồ thị**.

## 2. Các công trình theo hướng đồ thị

**Liu và cộng sự (2019), *Graph Convolution for Multimodal Information Extraction from Visually
Rich Documents*.** arXiv:1903.11279. Công trình nền tảng của hướng này: biểu diễn mỗi đoạn văn
bản thành một nút, dùng tích chập đồ thị để kết hợp thông tin văn bản với thông tin bố cục.

**Yu và cộng sự (2020), *PICK: Processing Key Information Extraction from Documents using
Improved Graph Learning-Convolutional Networks*.** ICPR 2020, arXiv:2004.07464. Kết hợp đặc
trưng văn bản và đặc trưng thị giác trong cùng một biểu diễn đồ thị, đồng thời học cả cấu trúc
đồ thị thay vì cố định trước. Đây là công trình gần nhất với hướng "kết hợp mạng tích chập và
đồ thị" của đề tài.

**Doc2Graph (2022), *a Task Agnostic Document Understanding Framework based on Graph Neural
Networks*.** arXiv:2208.11168, công bố tại hội thảo thuộc ECCV 2022. Xây dựng khung tổng quát
giải nhiều bài toán hiểu tài liệu bằng cùng một biểu diễn đồ thị, gồm cả hiểu biểu mẫu, phân
tích bố cục hóa đơn và phát hiện bảng.

**SEG-Doc (2025), *A simple yet efficient graph neural network framework for document key
information extraction*.** Tạp chí Neurocomputing. Đề xuất cơ chế chú ý theo bố cục khi truyền
thông điệp giữa các nút, đánh giá trên FUNSD, SROIE và RVL-CDIP.

## 3. Hướng tiền huấn luyện đa phương thức

**LayoutLMv3 (Huang và cộng sự, 2022), *Pre-training for Document AI with Unified Text and Image
Masking*.** Sử dụng nhúng theo mảng ảnh thay cho mạng tích chập làm khung xương, huấn luyện trước
với ba mục tiêu kết hợp văn bản và hình ảnh.

Hướng này cho kết quả cao nhưng chi phí huấn luyện lớn. Trong phạm vi đề tài, LayoutLMv3 được sử
dụng làm **mốc so sánh** thông qua kết quả đã công bố kèm bộ dữ liệu DocILE, không huấn luyện lại.

## 4. Bộ dữ liệu và mốc đánh giá

| Bộ dữ liệu | Quy mô | Đặc điểm |
|---|---|---|
| **DocILE** | 6.680 tài liệu có chú giải, 100.000 tài liệu tổng hợp | Hóa đơn và đơn hàng, 55 loại trường, hai bài toán KILE và LIR |
| FUNSD | 199 biểu mẫu | Biểu mẫu quét nhiễu, nhãn tổng quát |
| SROIE | Hóa đơn bán lẻ | Bốn trường thông tin |
| CORD | Hóa đơn bán lẻ | Nhiều trường hơn SROIE |
| RVL-CDIP | Phân loại tài liệu | Không phải bài toán trích xuất |

**Šimša và cộng sự (2023), *DocILE Benchmark for Document Information Localization and
Extraction*.** ICDAR 2023. Bộ dữ liệu chính của đề tài. Cung cấp sẵn kết quả OCR kèm tọa độ từng
từ và các mô hình nền để so sánh, gồm RoBERTa, LayoutLMv3 và DETR.

## 5. Vị trí của đề tài

Đề tài kế thừa hướng đồ thị của Liu và cộng sự và PICK, thực hiện trên bộ dữ liệu DocILE, với
hai điểm khác biệt so với các công trình đã khảo sát:

1. **Đo trần lý thuyết của phương pháp trước khi xây dựng mô hình.** Khảo sát cho thấy chỉ 95,64%
   số trường chú giải khớp được với từ do OCR trả về. Đây là giới hạn trên của mọi mô hình phân
   loại nút trên đồ thị từ, và là con số các công trình đã khảo sát không báo cáo.

2. **Bổ sung lớp kiểm chứng ở mức toàn tài liệu.** Các công trình theo hướng đồ thị dừng ở kết
   quả phân loại nút. Đề tài bổ sung một lớp suy luận kiểm tra tính nhất quán giữa các trường
   đã trích xuất, hiệu chỉnh mâu thuẫn, và đo riêng phần đóng góp của lớp này.

## 6. Cần đọc thêm

- Cách xử lý mất cân bằng nhãn trong bài toán phân loại nút, do phân bố loại trường trong DocILE
  rất lệch.
- Các phương án gom nhóm dòng hàng bằng dự đoán cạnh, phục vụ bài toán LIR.
- Phương pháp gán nhãn BIO trên đồ thị, do 26% số trường trải trên nhiều từ.
