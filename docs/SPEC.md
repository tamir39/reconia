# ĐẶC TẢ HỆ THỐNG

**Đề tài:** Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin
từ hóa đơn (ĐA161)

*Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction*

Tài liệu này đặc tả phạm vi, bài toán, kiến trúc và phương pháp thực nghiệm, theo đề tài nghiên cứu
đã được cập nhật trên hệ thống ELIT.

> **Trọng tâm đề tài:** Đánh giá, cài đặt và so sánh thực nghiệm 6 mô hình (GNN, RoBERTa,
> LayoutLMv3, LiLT và hai mô hình thị giác–ngôn ngữ; tên cụ thể đang chờ nhóm xác nhận) trên 3 bộ dữ liệu hóa đơn chuẩn (DocILE, CORD-v2, SROIE) dưới một khung đo lường thống nhất.

---

## 1. Bài toán

Đề tài thực hiện **đánh giá và so sánh các phương pháp học sâu** cho bài toán trích xuất thông
tin từ hóa đơn (Document Information Extraction), theo yêu cầu cốt lõi:

1. So sánh sáu mô hình: GNN, RoBERTa, LayoutLMv3, LiLT và hai VLM theo phạm vi đã thống nhất sau buổi họp. Hai VLM chưa được ghi nhận tên cụ thể; không coi là đã hiện thực.
2. Chuẩn hóa và thực nghiệm trên 3 bộ dữ liệu: DocILE (hóa đơn thương mại đa trang, phức tạp), CORD-v2 (biên lai ẩm thực, đa trường) và SROIE (biên lai bán lẻ).
3. Cài đặt hoặc huấn luyện lại các mô hình dưới **điều kiện thực nghiệm thống nhất** (cùng tập train/val, cùng chất lượng OCR, cùng pipeline tính chỉ số).
4. Đánh giá theo giá trị trường trên cả sáu mô hình; báo cáo riêng chỉ số BIO ở mức từ và chỉ số định vị chính thức của DocILE. Xem [EVALUATION.md](EVALUATION.md).
5. So sánh và phân tích ưu/nhược điểm, tốc độ suy luận (Latency ms) và thời gian huấn luyện/suy luận của từng phương pháp.

Bài toán trích xuất được định nghĩa theo chuẩn DocILE Benchmark:

| Bài toán | Tên viết tắt | Nội dung |
|---|---|---|
| **Key Information Localization and Extraction** | **KILE** | Xác định vị trí (hộp bao) và phân loại các trường thông tin chính trên tài liệu |
| **Line Item Recognition** | **LIR** | Gom các trường thông tin thành từng dòng hàng (hỗ trợ cho DocILE) |

**Đóng góp chính của đề tài là khung thực nghiệm thống nhất.** Kết quả công bố trong các công
trình khác nhau thường không so sánh trực tiếp được với nhau do khác biệt về cách chia tập dữ
liệu, cách tiền xử lý và cách tính chỉ số. Đề tài đưa các mô hình về cùng một điều kiện đo.

### 1.1. Các mô hình so sánh

| Nhóm phương pháp | Mô hình | Kiến trúc đại diện | Nơi huấn luyện |
|---|---|---|---|
| **Đồ thị (GNN)** | Relational GCN / Edge-augmented GNN | Đồ thị tài liệu đa quan hệ, nút là từ OCR (128-d text + 17-d layout), cạnh không gian ($r_{ij} \in \mathbb{R}^6$) | Môi trường cục bộ (Local GPU) |
| **Mô hình chuỗi** | RoBERTa-base | Pre-trained Transformer 1D bổ sung positional encoding 2D của hộp bao | Cloud GPU (Kaggle / Colab) |
| **Đa phương thức** | LayoutLMv3-base | Multimodal Transformer kết hợp Text, 2D Layout và Visual Patches | Cloud GPU (Kaggle / Colab) |
| **Đa phương thức nhẹ** | LiLT-base | Language-Independent Layout Transformer (tách biệt luồng văn bản và luồng bố cục) | Cloud GPU (Kaggle / Colab) |

| **Thị giác–ngôn ngữ** | VLM 1 — chờ xác nhận tên | Sinh các trường từ ảnh và chỉ dẫn | Chốt sau khi chọn kích thước mô hình |
| **Thị giác–ngôn ngữ** | VLM 2 — chờ xác nhận tên | Sinh các trường từ ảnh và chỉ dẫn | Chốt sau khi chọn kích thước mô hình |

Hiện có mã nguồn GNN và RoBERTa. LayoutLMv3, LiLT và hai VLM chưa hiện thực.

### 1.2. Môi trường huấn luyện

- **Môi trường cục bộ (Local):** GPU tiêu chuẩn hỗ trợ PyTorch CUDA. Dùng để phát triển, tiền xử lý dữ liệu, huấn luyện mô hình GNN và chạy khung đánh giá thống nhất.
- **Môi trường Cloud GPU:** GPU hỗ trợ CUDA (Kaggle/Colab). Dùng để fine-tune các mô hình Transformer tiền huấn luyện (RoBERTa, LayoutLMv3, LiLT) với checkpointing và early stopping định kỳ.

## 2. Dữ liệu thực nghiệm

Đề tài sử dụng 3 bộ dữ liệu mở chuẩn quốc tế, đại diện cho các hình thái tài liệu hóa đơn và biên lai khác nhau:

### 2.1. Bộ dữ liệu DocILE (ICDAR 2023)
- **Quy mô:** Tập `annotated-trainval` gồm 5.680 tài liệu có nhãn đầy đủ (`app/data/docile_splits.json`: 4.680 train, 500 val, 500 test local có nhãn) và tập `test_official` gồm 1.000 tài liệu phục vụ suy luận nộp bài thi ICDAR RRC.
- **Đặc trưng:** Hóa đơn thương mại doanh nghiệp thực tế (B2B), đa trang (77% 1 trang, 19% 2 trang, 4% 3 trang trở lên). Bố cục dạng bảng phức tạp, biến thiên cao.
- **Số loại trường:** 55 loại trường quan sát được (36 trường KILE và 19 trường LIR).
- **Thành phần:** PDF vector gốc, kết quả OCR có sẵn kèm hộp bao từng từ, nhãn chú giải chính xác.

### 2.2. Bộ dữ liệu CORD-v2 (Consolidated Receipt Dataset)
- **Quy mô:** 1.000 biên lai thanh toán (800 train, 100 val, 100 test).
- **Đặc trưng:** Biên lai dịch vụ ăn uống và bán lẻ (tiếng Anh/Indonesia), 1 trang, hình ảnh chụp thực tế.
- **Số loại trường:** 30 nhãn phân cấp chi tiết thuộc 4 nhóm chính: `menu`, `sub_total`, `total`, `void_menu`.
- **Thành phần:** Ảnh biên lai gốc, danh sách từ OCR kèm tọa độ đa giác, nhãn phân loại BIO.

### 2.3. Bộ dữ liệu SROIE (ICDAR 2019 Task 3)
- **Quy mô:** 626 biên lai bán lẻ (`app/data/sroie_splits.json`: 500 train, 63 val, 63 test ngẫu nhiên không thiên lệch với `seed=42`).
- **Đặc trưng:** Biên lai quét/chụp thực tế từ máy in nhiệt, nhiều nhiễu hình ảnh, góc nghiêng và mờ nhòe.
- **Số loại trường:** 4 trường thông tin then chốt: `company`, `date`, `address`, `total`.
- **Thành phần:** Ảnh JPG gốc, kết quả OCR từng dòng văn bản, nhãn thực thể chính xác.

### 2.4. Kết quả khảo sát và chuẩn hóa đầu vào chung
- **Giới hạn trần OCR:** Khảo sát trước đây trên toàn bộ tập train gốc của DocILE ghi nhận 94,42% trường khớp ít nhất một từ OCR. Đây là độ phủ của phép gán nhãn đang dùng, không phải trần F1 chính thức; cần đo lại theo cách chia tập hiện tại. Xem [DATA.md](DATA.md).
- **Ranh giới từ đa phần (Multi-word fields):** Khoảng 26% trường trên DocILE và đa số trường `address`/`company` trên SROIE trải dài trên nhiều từ liên tiếp. Do đó, tất cả dữ liệu được chuẩn hóa theo định dạng đánh nhãn chuỗi **BIO (Beginning - Inside - Outside)**.
- **Không gian tọa độ thống nhất:** Tất cả tọa độ hộp bao từ được chuẩn hóa kép:
  1. Tọa độ thực $[x_0, y_0, x_1, y_1]$ (pixel / PDF points) phục vụ trích xuất đặc trưng hình học đồ thị.
  2. Tọa độ chuẩn hóa tỉ lệ $[0, 1000]$ phục vụ nhúng không gian 2D cho Transformer (LayoutLMv3, LiLT).
- **Bảo toàn thứ tự đọc:** Pipeline tiền xử lý sắp xếp các từ theo thứ tự tự nhiên (từ trên xuống dưới, từ trái sang phải) trước khi xây dựng chuỗi token hoặc đồ thị.

## 3. Biểu diễn đồ thị tài liệu (Document Graph)

Đồ thị hóa đơn được xây dựng ở mức **từ OCR (word-level graph)** $G = (V, E, X, R)$:

### 3.1. Đặc trưng nút ($X \in \mathbb{R}^{|V| \times d}$)
Mỗi nút đại diện cho một từ do OCR trả về, gồm 2 thành phần chính:
1. **Ngữ nghĩa văn bản ($d_{text} = 128$):** Vector nhúng từ (FastText/Word2Vec hoặc learnable embedding được tiền huấn luyện trên hóa đơn).
2. **Hình học không gian ($d_{layout} = 17$):** Vector 17 chiều biểu diễn thuộc tính hình học và vị trí:
   $$x_{geom} = [x_0, y_0, x_1, y_1, w, h, x_c, y_c, w/h, \text{area}, \text{norm}_w, \text{norm}_h, \sin\theta, \cos\theta, \text{page}, \text{conf}, \text{is\_upper}]$$
   Tổng số chiều đặc trưng nút: $d = 128 + 17 = 145$ (ở pha mở rộng có thể ghép thêm $d_{visual} = 128$ trích từ ROI của ảnh).

### 3.2. Cấu trúc cạnh và quan hệ không gian ($E, R$)
Cạnh nối giữa các từ thể hiện cấu trúc ngữ cảnh bố cục hai chiều. Đề tài hỗ trợ 4 loại cạnh không gian (`edge_type`):
- `same_line`: Nối các từ nằm trên cùng dòng theo thứ tự ngang.
- `vertical_nearest`: Nối các từ thẳng hàng dọc gần nhất (trên - dưới), quan trọng cho các cột bảng hóa đơn.
- `knn`: Cạnh nối $k$ láng giềng gần nhất theo khoảng cách Euclid ($k=5$), **tuyệt đối chỉ nối các từ trên cùng một trang vật lý**.
- `self_loop`: Khuyên tự nối để duy trì thông tin nội tại của nút.

**Vector quan hệ không gian liên tục ($r_{ij} \in \mathbb{R}^6$):** Theo công trình nền tảng của Liu và cộng sự (2019), cạnh giữa nút $i$ và nút $j$ mang vector đặc trưng:
$$r_{ij} = \left[\frac{x_j - x_i}{w_i}, \frac{y_j - y_i}{h_i}, \frac{w_j}{w_i}, \frac{h_j}{h_i}, \frac{\text{dist}_{ij}}{h_i}, \text{angle}_{ij}\right]$$
Vector $r_{ij}$ cung cấp thông tin hình học liên tục cho cơ chế truyền thông điệp (Message Passing) của GNN.

## 4. Bốn mô hình học sâu so sánh

### 4.1. Mô hình 1: Đồ thị tài liệu (GNN)
- **Kiến trúc:** Relational Graph Convolutional Network (RGCN) hoặc Edge-augmented GAT (Graph Attention Network).
- **Cơ chế:** Cập nhật biểu diễn nút qua nhiều tầng lan truyền thông tin theo các loại quan hệ không gian (`edge_type`) và vector quan hệ $r_{ij}$.
- **Phân loại:** Bộ phân loại MLP dự đoán nhãn BIO trên từng nút từ vector biểu diễn sau GNN. Huấn luyện trên GPU cục bộ (kiến trúc gọn nhẹ, tối ưu bộ nhớ).

### 4.2. Mô hình 2: Chuỗi văn bản (RoBERTa-base + Layout Positional Embedding)
- **Kiến trúc:** RoBERTa-base với bộ mã hóa 12 tầng Transformer.
- **Cơ chế:** Bổ sung nhúng tọa độ 2D của hộp bao $[x_0, y_0, x_1, y_1]$ vào vector nhúng từ 1D. Đóng vai trò mốc so sánh (baseline) cho trường phái tuần tự không dùng hình ảnh.

### 4.3. Mô hình 3: Đa phương thức hoàn chỉnh (LayoutLMv3-base)
- **Kiến trúc:** Multimodal Transformer kết hợp đồng thời Văn bản (BPE token), Bố cục 2D ($[x_0, y_0, x_1, y_1]$ chuẩn hóa $[0, 1000]$) và Thị giác (Visual Patch Embeddings qua Linear Projection).
- **Cơ chế:** Cơ chế chú ý đa phương thức hợp nhất (Unified Multi-modal Pre-training), đại diện cho giải pháp công nghệ mạnh mẽ nhất hiện nay trong Document AI.

### 4.4. Mô hình 4: Đa phương thức tách biệt (LiLT-base)
- **Kiến trúc:** Language-Independent Layout Transformer.
- **Cơ chế:** Sử dụng hai luồng xử lý độc lập cho văn bản và bố cục, kết hợp qua cơ chế Cross-Attention hai chiều. Nhẹ hơn LayoutLMv3 và dễ dàng chuyển giao ngôn ngữ.

---

## 5. Khung thực nghiệm thống nhất

Đây là nội dung trọng tâm của đề tài. Mọi mô hình được đo dưới cùng một điều kiện:

| Yếu tố | Quy định thống nhất |
|---|---|
| **Dữ liệu** | 3 bộ dữ liệu: DocILE, CORD-v2, SROIE qua các Dataset Adapter chung |
| **Cách chia tập** | Cố định ngẫu nhiên với seed `42` (`docile_splits.json`, `sroie_splits.json`, CORD official split) |
| **Tiền xử lý** | Dùng chung một kết quả OCR, cùng chuẩn hóa nhãn BIO và thứ tự đọc tự nhiên |
| **Chỉ số đánh giá** | Precision, Recall, F1 ở cấp độ thực thể (Exact Match) và độ trễ suy luận Latency (ms) đo đạc độc lập trên tập Test |
| **Khả năng tái lập** | Ghi nhận chi tiết thông số phần cứng, số lượng tham số mô hình, thời gian huấn luyện và độ trễ suy luận (ms) |

Khung đánh giá được thiết kế **độc lập với kiến trúc mô hình**: bổ sung một mô hình mới chỉ yêu
cầu cung cấp đầu ra theo định dạng chung, không sửa phần đánh giá.

Nội dung báo cáo kết quả:
1. Bảng so sánh các mô hình trên 3 bộ dữ liệu (F1 tổng quát và theo từng nhóm trường).
2. Phân tích chi phí tính toán: số lượng tham số mô hình, thời gian huấn luyện, tốc độ suy luận (Latency ms).
3. So sánh định tính: phân tích các trường hợp lỗi điển hình của từng mô hình (lệch dòng, đa cột, bảng biểu phức tạp).
4. Hệ thống báo cáo tự động (`app.evaluation.report`): Tự động xuất báo cáo Markdown, số liệu CSV, JSON và biểu đồ trực quan PNG sau khi hoàn thành thực nghiệm. Lệnh tổng hợp độc lập: `python -m app.evaluation.report --report-dir reports`.

## 6. Quy ước kỹ thuật bắt buộc

1. **Hạt giống ngẫu nhiên cố định:** Tất cả mã nguồn khởi tạo tham số và chia tập phải cố định `seed = 42`.
2. **Tính di động của đường dẫn:** Không hardcode đường dẫn tuyệt đối; luôn truy cập qua `app.data.paths` hoặc biến môi trường `DOCILE_DIR`, `SROIE_DIR`, `CORD_DIR`.
3. **Độ tin cậy và gán nhãn:** Các dự đoán phải giữ lại điểm xác suất (confidence score) để hỗ trợ phân tích ngưỡng tin cậy.
4. **Quản lý bộ nhớ GPU:** Giải phóng bộ nhớ (`torch.cuda.empty_cache()`) và checkpointing định kỳ sau mỗi epoch trên môi trường giới hạn như Kaggle.

---

## 7. Lớp Giao Diện Web Minh Họa (Web Demo Layer)

Bên cạnh khung thực nghiệm dòng lệnh (CLI), dự án có khung ứng dụng web minh họa (`app/main.py`, `app/ui/routes.py`, `app/models/`):
- **Công nghệ nền tảng:** FastAPI, Jinja2 template, HTMX cho tương tác bất đồng bộ không cần SPA phức tạp, và CSS cục bộ cho giao diện hiển thị.
- **Chưa hiện thực — trực quan hóa tài liệu và OCR:** Dự kiến hiển thị trực quan trang tài liệu hóa đơn/biên lai cùng hệ thống hộp bao tọa độ từ OCR, phân biệt trực quan các vùng trường thông tin.
- **Chưa hiện thực — đối chiếu thực thể trích xuất:** Hỗ trợ đối chiếu trực quan kết quả dự đoán của từng mô hình học sâu với nhãn thực tế (ground-truth), phục vụ công tác nghiệm thu và phân tích định tính các ca lỗi (Error Analysis).
- **Quản trị và tra cứu chứng từ:** Sử dụng SQLAlchemy ORM và Alembic migration quản lý cơ sở dữ liệu chứng từ, hỗ trợ tìm kiếm và lọc theo loại chứng từ.

