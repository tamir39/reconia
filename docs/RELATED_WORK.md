# TỔNG QUAN CÔNG TRÌNH LIÊN QUAN

**Đề tài:** Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin
từ hóa đơn (ĐA161)

Tài liệu tổng hợp các công trình liên quan đến trích xuất thông tin từ tài liệu nghiệp vụ, phục
vụ việc lựa chọn các mô hình đưa vào so sánh.

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

Đề tài lựa chọn mô hình từ cả ba hướng trên để so sánh, thay vì tập trung vào một hướng duy nhất.

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
Masking*.** ACM Multimedia 2022. Sử dụng nhúng theo mảng ảnh (patch embeddings) thay cho mạng tích chập làm khung xương thị giác, kết hợp đồng thời văn bản, bố cục 2D và hình ảnh dưới một cơ chế Attention chung.

**LiLT (Wang và cộng sự, 2022), *LiLT: A Language-Independent Layout Transformer for Structured Document Information Extraction*.** ACL 2022. Đề xuất kiến trúc tách biệt luồng xử lý ngôn ngữ và bố cục, cho phép tận dụng bố cục độc lập với ngôn ngữ văn bản và giảm nhẹ đáng kể chi phí tính toán so với LayoutLMv3.

Trong phạm vi đề tài, các mô hình Transformer tiền huấn luyện (RoBERTa, LayoutLMv3, LiLT) được fine-tune trực tiếp trên Kaggle GPU trên cùng các tập dữ liệu, sau đó được đối chiếu trực tiếp với mô hình Đồ thị (GNN) trên cùng một khung đo lường thống nhất.

## 4. Bộ dữ liệu và mốc đánh giá

| Bộ dữ liệu | Quy mô | Đặc điểm | Vai trò trong đề tài |
|---|---|---|---|
| **DocILE** | 5.680 tài liệu train/val | Hóa đơn & đơn hàng B2B thực tế, đa trang, 55 loại trường | Bộ dữ liệu chính (thực nghiệm KILE & LIR) |
| **CORD-v2** | 1.000 biên lai | Biên lai ẩm thực & dịch vụ, 30 loại nhãn phân cấp | Bộ dữ liệu mở rộng (thực nghiệm KILE) |
| **SROIE** | 987 biên lai | Biên lai in nhiệt thực tế, 4 trường chính (company, date, address, total) | Bộ dữ liệu mở rộng (thực nghiệm KILE) |
| FUNSD | 199 biểu mẫu | Biểu mẫu quét nhiễu, nhãn tổng quát | Tham khảo |
| RVL-CDIP | 400.000 tài liệu | Phân loại tài liệu | Tiền huấn luyện (tham khảo) |

**Šimša và cộng sự (2023), *DocILE Benchmark for Document Information Localization and
Extraction*.** ICDAR 2023. Bộ dữ liệu trung tâm của đề tài. Cung cấp sẵn kết quả OCR kèm tọa độ từng
từ và các mô hình nền để so sánh, gồm RoBERTa, LayoutLMv3 và DETR.

## 5. Vị trí của đề tài

Đề tài không đề xuất kiến trúc mới. Đề tài thực hiện **đánh giá và so sánh** các phương pháp đã
công bố dưới một điều kiện thực nghiệm thống nhất, với hai điểm cần nhấn mạnh:

1. **Kết quả công bố giữa các công trình thường không so sánh trực tiếp được với nhau.** Các
   công trình khác nhau về cách chia tập dữ liệu, chất lượng OCR đầu vào và cách tính chỉ số.
   Việc đưa các mô hình về cùng một điều kiện đo là nội dung chính của đề tài.

2. **Đo trần lý thuyết của nhóm phương pháp gán nhãn theo từ.** Khảo sát cho thấy chỉ 95,64% số
   trường chú giải khớp được với từ do OCR trả về. Đây là giới hạn trên của mọi mô hình gán nhãn
   ở mức từ. Con số này cần được nêu khi so sánh nhóm phương pháp gán nhãn theo từ với nhóm
   phương pháp sinh văn bản trực tiếp, vì hai nhóm không chịu chung giới hạn này.

## 6. Cần đọc thêm

- Cách xử lý mất cân bằng nhãn trong bài toán phân loại nút, do phân bố loại trường trong DocILE
  rất lệch.
- Các phương án gom nhóm dòng hàng bằng dự đoán cạnh, phục vụ bài toán LIR.
- Phương pháp gán nhãn BIO trên đồ thị, do 26% số trường trải trên nhiều từ.
