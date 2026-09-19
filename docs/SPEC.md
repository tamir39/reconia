# ĐẶC TẢ HỆ THỐNG

**Đề tài:** Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin
từ hóa đơn (ĐA161)

*Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction*

Tài liệu này đặc tả phạm vi, bài toán, kiến trúc và phương pháp thực nghiệm, theo đề tài nghiên cứu
đã được cập nhật trên hệ thống ELIT.

> **Trọng tâm đề tài:** Đánh giá, cài đặt và so sánh thực nghiệm 4 mô hình học sâu (GNN, RoBERTa,
> LayoutLMv3, LiLT) trên 3 bộ dữ liệu hóa đơn chuẩn (DocILE, CORD-v2, SROIE) dưới một khung đo lường thống nhất.
> Các nội dung liên quan đến ứng dụng đối soát nghiệp vụ (Sections 5-6) thuộc về lớp demo mở rộng đã đóng băng.

---

## 1. Bài toán

Đề tài thực hiện **đánh giá và so sánh các phương pháp học sâu** cho bài toán trích xuất thông
tin từ hóa đơn (Document Information Extraction), theo yêu cầu cốt lõi:

1. Khảo sát và lựa chọn 4 mô hình học sâu đại diện cho các trường phái tiếp cận: Đồ thị (GNN), Mô hình chuỗi văn bản (RoBERTa), Đa phương thức hoàn chỉnh (LayoutLMv3) và Đa phương thức tách biệt (LiLT).
2. Chuẩn hóa và thực nghiệm trên 3 bộ dữ liệu: DocILE (hóa đơn thương mại đa trang, phức tạp), CORD-v2 (biên lai ẩm thực, đa trường) và SROIE (biên lai bán lẻ).
3. Cài đặt hoặc huấn luyện lại các mô hình dưới **điều kiện thực nghiệm thống nhất** (cùng tập train/val, cùng chất lượng OCR, cùng pipeline tính chỉ số).
4. Đánh giá bằng Precision, Recall và F1-score (cả micro-F1 và macro-F1) ở cấp độ thực thể (entity-level).
5. So sánh và phân tích ưu/nhược điểm, tốc độ suy luận, chi phí tính toán (GPU memory, thời gian train/inference) của từng phương pháp.

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
| **Đồ thị (GNN)** | Relational GCN / Edge-augmented GNN | Đồ thị tài liệu đa quan hệ, nút là từ OCR (128-d text + 17-d layout), cạnh không gian ($r_{ij} \in \mathbb{R}^6$) | Máy cá nhân (GTX 1650 Ti) |
| **Mô hình chuỗi** | RoBERTa-base | Pre-trained Transformer 1D bổ sung positional encoding 2D của hộp bao | Kaggle GPU (T4 / P100) |
| **Đa phương thức** | LayoutLMv3-base | Multimodal Transformer kết hợp Text, 2D Layout và Visual Patches | Kaggle GPU (T4 / P100) |
| **Đa phương thức nhẹ** | LiLT-base | Language-Independent Layout Transformer (tách biệt luồng văn bản và luồng bố cục) | Kaggle GPU (T4 / P100) |

### 1.2. Môi trường huấn luyện

- **Máy cá nhân:** Intel Core i7 / AMD Ryzen, GPU NVIDIA GeForce GTX 1650 Ti (4GB VRAM), PyTorch 2.0+ CUDA. Dùng để phát triển, tiền xử lý dữ liệu, huấn luyện mô hình GNN và chạy khung đánh giá thống nhất.
- **Kaggle GPU:** T4 x 2 (16GB VRAM) hoặc P100 (16GB VRAM). Dùng để fine-tune các mô hình Transformer tiền huấn luyện (RoBERTa, LayoutLMv3, LiLT) với checkpointing và early stopping định kỳ.

## 2. Dữ liệu thực nghiệm

Đề tài sử dụng 3 bộ dữ liệu mở chuẩn quốc tế, đại diện cho các hình thái tài liệu hóa đơn và biên lai khác nhau:

### 2.1. Bộ dữ liệu DocILE (ICDAR 2023)
- **Quy mô:** Tập `annotated-trainval` gồm 5.680 tài liệu (5.180 train, 500 val).
- **Đặc trưng:** Hóa đơn thương mại doanh nghiệp thực tế (B2B), đa trang (77% 1 trang, 19% 2 trang, 4% 3 trang trở lên). Bố cục dạng bảng phức tạp, biến thiên cao.
- **Số loại trường:** 55 loại trường (36 trường KILE và 18 trường LIR).
- **Thành phần:** PDF vector gốc, kết quả OCR có sẵn kèm hộp bao từng từ, nhãn chú giải chính xác.

### 2.2. Bộ dữ liệu CORD-v2 (Consolidated Receipt Dataset)
- **Quy mô:** 1.000 biên lai thanh toán (800 train, 100 val, 100 test).
- **Đặc trưng:** Biên lai dịch vụ ăn uống và bán lẻ (tiếng Anh/Indonesia), 1 trang, hình ảnh chụp thực tế.
- **Số loại trường:** 30 nhãn phân cấp chi tiết thuộc 4 nhóm chính: `menu`, `sub_total`, `total`, `void_menu`.
- **Thành phần:** Ảnh biên lai gốc, danh sách từ OCR kèm tọa độ đa giác, nhãn phân loại BIO.

### 2.3. Bộ dữ liệu SROIE (ICDAR 2019 Task 3)
- **Quy mô:** 987 biên lai bán lẻ (626 train, 361 test). Để phục vụ thực nghiệm thống nhất, tập train được chia ngẫu nhiên cố định (seed 42) thành 500 train và 126 val (`app/data/sroie_splits.json`).
- **Đặc trưng:** Biên lai quét/chụp thực tế từ máy in nhiệt, nhiều nhiễu hình ảnh, góc nghiêng và mờ nhòe.
- **Số loại trường:** 4 trường thông tin then chốt: `company`, `date`, `address`, `total`.
- **Thành phần:** Ảnh JPG gốc, kết quả OCR từng dòng văn bản, nhãn thực thể chính xác.

### 2.4. Kết quả khảo sát và chuẩn hóa đầu vào chung
- **Giới hạn trần OCR:** Mức độ khớp giữa trường chú giải và từ OCR đạt ~95.6% trên DocILE, ~98-100% trên SROIE. Đây là giới hạn lý thuyết trên đối với các mô hình phân loại nhãn nút/token.
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
- **Phân loại:** Bộ phân loại MLP dự đoán nhãn BIO trên từng nút từ vector biểu diễn sau GNN. Huấn luyện cục bộ trên GPU GTX 1650 Ti (bộ nhớ nhẹ, $< 1$GB VRAM).

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

> [!NOTE]
> ### Lưu ý về Lớp Ứng dụng Đối soát & Agent (Mục 5 & 6)
> Các nội dung tại **Mục 5 (Agent kiểm chứng)** và **Mục 6 (Ứng dụng đối soát)** thuộc về hệ thống đối soát 3 chiều (3-way matching demo) đã được phát triển trong giai đoạn trước. Nhằm tập trung tối đa nguồn lực vào đề tài nghiên cứu học thuật về Học sâu, **toàn bộ lớp ứng dụng này đã được đóng băng (frozen)** và giữ lại trong kho mã nguồn làm tài liệu tham khảo cho việc tích hợp vào hệ thống nghiệp vụ sau này.

## 5. Lớp Agent kiểm chứng (Demo Layer — Frozen)

Mô hình đồ thị phân loại từng nút một cách cục bộ và không có khái niệm về tính nhất quán ở mức
toàn tài liệu. Lớp Agent xử lý đúng nhóm sai sót này:
- Phát hiện xung đột giữa các trường thông tin đơn nhất.
- Kiểm tra tính toán số học giữa tổng tiền và chi tiết các dòng hàng.
- Chi tiết kiến trúc xem tại [AGENT.md](AGENT.md).

## 6. Lớp ứng dụng minh họa đối soát (Demo Layer — Frozen)

Ứng dụng đối soát chứng từ 3 chiều (Hóa đơn - Phiếu xuất/nhập kho - Đơn đặt hàng) sử dụng kết quả
trích xuất từ mô hình để minh họa khả năng ứng dụng thực tế. Chi tiết lược đồ cơ sở dữ liệu xem tại [ERD.md](ERD.md).

---

## 7. Khung thực nghiệm thống nhất

Đây là nội dung trọng tâm của đề tài. Mọi mô hình được đo dưới cùng một điều kiện:

| Yếu tố | Quy định thống nhất |
|---|---|
| **Dữ liệu** | 3 bộ dữ liệu: DocILE, CORD-v2, SROIE qua các Dataset Adapter chung |
| **Cách chia tập** | Cố định ngẫu nhiên với seed `42` (DocILE giữ nguyên tập gốc, SROIE theo `sroie_splits.json`, CORD-v2 theo official split) |
| **Tiền xử lý** | Dùng chung một kết quả OCR, cùng chuẩn hóa nhãn BIO và thứ tự đọc tự nhiên |
| **Chỉ số đánh giá** | Precision, Recall, Micro-F1, Macro-F1 tính ở cấp độ thực thể (entity-level matching) |
| **Khả năng tái lập** | Ghi nhận chi tiết thông số phần cứng, dung lượng VRAM, thời gian huấn luyện và tốc độ suy luận (ms/trang) |

Khung đánh giá được thiết kế **độc lập với kiến trúc mô hình**: bổ sung một mô hình mới chỉ yêu
cầu cung cấp đầu ra theo định dạng chung, không sửa phần đánh giá.

Nội dung báo cáo kết quả:
1. Bảng so sánh 4 mô hình trên 3 bộ dữ liệu (F1 tổng quát và theo từng nhóm trường).
2. Phân tích chi phí tính toán: số lượng tham số, thời gian train, bộ nhớ GPU tiêu thụ.
3. So sánh định tính: phân tích các trường hợp lỗi điển hình của từng mô hình (lệch dòng, đa cột, bảng biểu phức tạp).
4. Hệ thống báo cáo tự động (`app.evaluation.report`): Tự động xuất tệp Markdown, CSV, JSON và biểu đồ trực quan PNG (300 DPI) sau khi huấn luyện. Hỗ trợ lệnh tổng hợp độc lập: `python -m app.evaluation.report --reports-dir reports`.

## 8. Quy ước kỹ thuật bắt buộc

1. **Hạt giống ngẫu nhiên cố định:** Tất cả mã nguồn khởi tạo tham số và chia tập phải cố định `seed = 42`.
2. **Tính di động của đường dẫn:** Không hardcode đường dẫn tuyệt đối; luôn truy cập qua `app.data.paths` hoặc biến môi trường `DOCILE_DIR`, `SROIE_DIR`, `CORD_DIR`.
3. **Độ tin cậy và gán nhãn:** Các dự đoán phải giữ lại điểm xác suất (confidence score) để hỗ trợ phân tích ngưỡng tin cậy.
4. **Quản lý bộ nhớ GPU:** Giải phóng bộ nhớ (`torch.cuda.empty_cache()`) và checkpointing định kỳ sau mỗi epoch trên môi trường giới hạn như Kaggle.
