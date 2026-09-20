# Reconia: Đánh Giá và So Sánh Các Phương Pháp Học Sâu Cho Trích Xuất Thông Tin Từ Hóa Đơn

*Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction*

**Đồ án môn:** Dự án CNTT (ĐA161) — HK1/2026-2027  
**Nhóm thực hiện:** Best Stuffs  
* Phí Vương Tường Tâm (MSSV: 52300155)  
* Vũ Hưng Thịnh (MSSV: 52300159)  
**Giảng viên hướng dẫn:** ThS. Dung Cẩm Quang  

---

## 1. Giới thiệu & Mục tiêu Nghiên cứu

Trích xuất thông tin từ tài liệu giàu bố cục thị giác (**Visually Rich Documents - VRDs**) như hóa đơn doanh nghiệp và biên lai thanh toán là bài toán quan trọng trong tự động hóa quy trình nghiệp vụ (Document AI).

Mục tiêu chính của đề tài là **thiết lập một khung thực nghiệm thống nhất (Unified Benchmark)** nhằm khảo sát, cài đặt và so sánh khách quan các hướng tiếp cận học sâu tiêu biểu cho bài toán:
1. **KILE (Key Information Localization and Extraction)**: Xác định vị trí và phân loại các trường thông tin chính (tên công ty, ngày tháng, tổng tiền, thuế, số hóa đơn,...).
2. **LIR (Line Item Recognition)**: Trích xuất và gom nhóm các trường chi tiết thành từng dòng hàng hoàn chỉnh trong bảng sản phẩm.

### Bốn mô hình học sâu đại diện cho 4 hướng tiếp cận:
1. **Mô hình Đồ thị (Graph-based)**: Mạng tích chập đồ thị quan hệ không gian (*Spatial Relational GCN*) tự hiện thực bằng PyTorch thuần, kết hợp Text Semantics + 17 đặc trưng hình học và vector quan hệ không gian cạnh (dựa trên [Liu et al., NAACL 2019](file:///d:/Thinh/tdt/Y4/DACNTT/papers/files/205) và [PICK, ICPR 2020](file:///d:/Thinh/tdt/Y4/DACNTT/papers/files/209)). Huấn luyện trên GPU cá nhân (GTX 1650 Ti 4GB).
2. **Mô hình Ngôn ngữ tuần tự 1D (Text-only Sequence LM)**: RoBERTa token classification (chuẩn baseline của [DocILE Benchmark, ICDAR 2023](file:///d:/Thinh/tdt/Y4/DACNTT/papers/files/199)).
3. **Mô hình Đa phương thức 2D (Multimodal 2D Transformer)**: LayoutLMv3 kết hợp đồng thời Text Tokens + 2D Bounding Boxes + Image Patches ([Huang et al., ACM MM 2022](file:///d:/Thinh/tdt/Y4/DACNTT/papers/files/218)).
4. **Mô hình Bố cục tách rời (Decoupled Layout Transformer)**: LiLT tách riêng nhánh ngôn ngữ và nhánh bố cục 2D độc lập ([Wang et al., ACL 2022](file:///d:/Thinh/tdt/Y4/DACNTT/papers/files/220)).

### Ba bộ dữ liệu thực nghiệm thống nhất:
* **DocILE**: 5.680 hóa đơn doanh nghiệp thực tế, đa trang (1-3 trang), 55 loại trường (KILE 36 loại, LIR 18 loại).
* **CORD-v2**: 1.000 hóa đơn nhà hàng chụp thực tế (800 train, 100 val, 100 test), 30 danh mục trường và gom nhóm dòng hàng.
* **SROIE**: 626 hóa đơn quét (500 train, 63 val, 63 test ngẫu nhiên không thiên lệch với `seed=42`), 4 trường khóa (`company`, `date`, `address`, `total`).

---

## 2. Cấu trúc Kho Mã Nguồn

```
reconia/
├── app/
│   ├── data/                 # Tầng chuẩn hóa đầu vào chung (Dataset Adapters & Schema)
│   │   ├── schema.py         # Kiểu dữ liệu trung gian IR: BBox, Word, Field, Document, DatasetAdapter
│   │   ├── docile.py         # DocileAdapter (tự động phân giải train, val, và test sau này)
│   │   ├── cord.py           # CordAdapter (chuẩn hóa 30 thực thể và line_item_id)
│   │   ├── sroie.py          # SroieAdapter (khớp nhãn token/line không tràn, seed=42)
│   │   ├── paths.py          # Cơ chế tự động dò tìm đường dẫn dữ liệu di động
│   │   └── sroie_splits.json # Phân chia ngẫu nhiên 500/63/63 tái lập được của SROIE
│   │
│   ├── graph/                # Tầng xây dựng đồ thị tài liệu (Document Graph)
│   │   ├── builder.py        # Dựng DocumentGraph, trích xuất 17 đặc trưng hình học nút
│   │   ├── edges.py          # Nối cạnh không gian (kNN theo trang, cùng dòng, cùng cột) & vector r_ij
│   │   ├── labels.py         # Gán nhãn BIO, tính thống kê độ phủ (ceiling) và giải mã thực thể
│   │   ├── cache.py          # Bộ đệm lưu đồ thị (.npz) và từ điển nhãn (.json)
│   │   └── prepare.py        # CLI tiền xử lý và sinh cache đồ thị cho các bộ dữ liệu
│   │
│   ├── models_dl/            # Các kiến trúc mô hình học sâu
│   │   └── gnn.py            # Mô hình 1: Spatial Relational GNN
│   ├── training/             # Vòng lặp huấn luyện, checkpointing, profiling tài nguyên
│   │   └── train_gnn.py      # Pipeline huấn luyện GNN trên SROIE, CORD, DocILE
│   ├── evaluation/           # Khung đo đạc Token/Entity F1, Resource Profiler & Trực quan hóa
│   │   ├── metrics.py        # Token Macro/Micro-F1, Entity Micro-F1, Latency & Peak VRAM Profiler
│   │   └── report.py         # Kết xuất báo cáo Markdown, CSV, JSON, Biểu đồ PNG & So sánh chéo
│   │
│   ├── models/               # [Đóng băng] Lược đồ CSDL PostgreSQL cho web demo
│   ├── ui/                   # [Đóng băng] Giao diện web đối soát demo (Jinja2 + HTMX)
│   └── main.py               # [Đóng băng] FastAPI app cho web demo
│
├── reports/                  # Báo cáo thực nghiệm đọc được & Biểu đồ trực quan hóa
│   ├── figures/              # Thư mục chứa biểu đồ huấn luyện và so sánh chéo (.png 300 DPI)
│   ├── benchmark_summary.md  # Bảng tổng hợp so sánh toàn diện tất cả mô hình trên các dataset
│   └── *_report.md           # Báo cáo chi tiết từng mô hình (kèm bảng per-class breakdown)
│
├── docs/                     # Tài liệu thiết kế và nghiên cứu học thuật
│   ├── SPEC.md               # Đặc tả bài toán, phương pháp, dữ liệu và khung thực nghiệm
│   ├── RELATED_WORK.md       # Tổng quan các công trình nghiên cứu liên quan
│   ├── AGENT.md              # [Đóng băng] Ghi chú thiết kế Agent kiểm chứng cho web demo
│   └── ERD.md                # [Đóng băng] Lược đồ CSDL quan hệ cho web demo
│
├── tests/                    # Bộ kiểm thử tự động toàn diện (57 unit tests)
├── requirements.txt          # Danh sách thư viện phụ thuộc của dự án
├── .env.example              # Mẫu biến môi trường cấu hình đường dẫn và CSDL
└── POLICY.md                 # Quy ước đóng góp git, đặt tên nhánh và commit
```

---

## 3. Cài Đặt & Hướng Dẫn Sử Dụng

### 3.1. Thiết lập Môi trường Ảo (Virtualenv)

Khuyến nghị sử dụng **Python 3.10**:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cơ bản và xử lý dữ liệu
pip install -r requirements.txt

# (Khuyến nghị) Cài đặt PyTorch có hỗ trợ GPU NVIDIA CUDA (cho GTX 1650 Ti):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3.2. Cấu hình Đường dẫn Dữ liệu (`.env`)

Mặc định, hệ thống tự động dò tìm các thư mục dữ liệu ngang hàng trong không gian làm việc (`../docile`, `../cord`, `../sroie`). Nếu bạn đặt dữ liệu ở vị trí khác (vd: trên Kaggle hoặc ổ đĩa khác), cấu hình trong tệp `.env`:

```bash
cp .env.example .env
```

Điền các biến tương ứng nếu cần tùy chỉnh:
```env
DOCILE_DIR=../docile/data/annotated-trainval
CORD_DIR=../cord
SROIE_DIR=../sroie/data
```

### 3.3. Chạy Kiểm Thử Tự Động (Automated Tests)

Chạy bộ kiểm thử tự động xác minh toàn bộ adapter dữ liệu, cấu trúc đồ thị, logic gán nhãn BIO và các tính năng hệ thống:

```bash
pytest tests/ -v
```

### 3.4. Chuẩn bị Dữ liệu & Sinh Từ Điển Nhãn (`prepare.py`)

Chạy công cụ dòng lệnh để xây dựng từ điển nhãn BIO và sinh bộ đệm đồ thị (`.cache/`):

```bash
# Với bộ dữ liệu SROIE (500 train, 63 val):
python -m app.graph.prepare --dataset sroie --splits train val

# Với bộ dữ liệu CORD-v2 (800 train, 100 validation):
python -m app.graph.prepare --dataset cord --splits train validation

# Với bộ dữ liệu DocILE (5.180 train, 500 val):
python -m app.graph.prepare --dataset docile --splits train val
```

### 3.5. Huấn Luyện Mô Hình Học Sâu & Tự Động Xuất Báo Cáo

Hệ thống hỗ trợ huấn luyện độc lập từng kiến trúc mô hình học sâu. Mọi script đều tự động tính toán loss có trọng số cân bằng lớp (`compute_class_weights`), profiling tài nguyên GPU VRAM, đo độ trễ suy luận, và ghi log song song ra console lẫn file log timestamp (`reports/train_<model>_<dataset>_<timestamp>.log`):

**Huấn luyện Mô hình 1: GNN (Spatial Relational Graph Neural Network)**:
```bash
# Huấn luyện GNN trên SROIE (kèm Early Stopping sau 5 epochs không tăng)
python -m app.training.train_gnn --dataset sroie --epochs 20 --patience 5

# Huấn luyện tiếp tục từ checkpoint đã lưu (Resume Training)
python -m app.training.train_gnn --dataset sroie --epochs 25 --resume checkpoints/gnn_sroie_latest.pt

# Huấn luyện GNN trên CORD-v2 theo dõi Entity F1
python -m app.training.train_gnn --dataset cord --epochs 15 --monitor-metric ent_f1

# Huấn luyện GNN trên DocILE (subset hoặc full)
python -m app.training.train_gnn --dataset docile --epochs 10 --limit-train 500 --limit-val 100
```

**Huấn luyện Mô hình 2: RoBERTa (RoBERTa-base + 2D Layout Positional Embedding)**:
```bash
# Huấn luyện RoBERTa trên SROIE (learning rate chuẩn fine-tuning 3e-5)
python -m app.training.train_roberta --dataset sroie --epochs 10 --patience 5 --batch-size 4

# Huấn luyện RoBERTa trên CORD-v2
python -m app.training.train_roberta --dataset cord --epochs 10 --patience 5 --batch-size 4

# Huấn luyện RoBERTa trên DocILE
python -m app.training.train_roberta --dataset docile --epochs 10 --limit-train 500 --limit-val 100
```

**Các tính năng huấn luyện nâng cao:**
- `--patience <int>` (hoặc `--early-stopping`): Tự động dừng sớm nếu chỉ số theo dõi không cải thiện sau $N$ epochs liên tiếp, bảo vệ GPU và tránh overfitting.
- `--resume <path>`: Nạp lại toàn bộ trọng số, optimizer, epoch và lịch sử huấn luyện từ tệp `.pt` để học tiếp tục liền mạch.
- `--monitor-metric <tok_macro|ent_f1|val_loss>`: Lựa chọn chỉ số giám sát để lưu mô hình tối ưu nhất (mặc định: `tok_macro`).
- `--clip-grad-norm <float>`: Cắt tỉa gradient (mặc định: `1.0`) giúp ổn định quá trình tối ưu hóa, triệt tiêu nguy cơ nổ gradient.
- **Cơ chế lưu Checkpoint kép:** Hệ thống luôn tự động lưu song song `<model>_<dataset>_best.pt` (checkpoint tối ưu nhất) và `<model>_<dataset>_latest.pt` (checkpoint epoch gần nhất phục vụ resume mọi lúc).

**Cơ chế xuất kết quả tự động khi hoàn thành:**
Khi quá trình huấn luyện kết thúc, hệ thống sẽ **tự động kết xuất trọn bộ báo cáo đọc được** vào thư mục `reports/`:
- `reports/<model>_<dataset>_report.md`: Báo cáo Markdown đọc được trực tiếp, trình bày bảng tóm tắt kết quả tốt nhất, bảng chi tiết từng trường thông tin (Precision, Recall, F1, Support) và bảng theo dõi từng epoch.
- `reports/<model>_<dataset>_history.csv`: Bảng tính chi tiết từng epoch, tương thích mở bằng Microsoft Excel hoặc Google Sheets.
- `reports/<model>_<dataset>_results.json`: Dữ liệu có cấu trúc phục vụ phân tích và tổng hợp tự động.
- `reports/figures/<model>_<dataset>_training_curves.png`: Biểu đồ 4-trong-1 chất lượng cao (300 DPI) gồm đường cong Loss, Token F1, Entity Micro-F1 và biểu đồ cột F1 của từng trường nhãn.
- `reports/benchmark_summary.md` & `reports/figures/benchmark_comparison.png`: Tự động cập nhật bảng so sánh đối đầu và biểu đồ so sánh chéo đa mô hình.


### 3.6. Công Cụ Tổng Hợp Báo Cáo & Vẽ Biểu Đồ So Sánh (`report.py`)

Khi bạn muốn tổng hợp lại toàn bộ kết quả, cập nhật bảng so sánh đối đầu hoặc vẽ lại biểu đồ so sánh chéo giữa tất cả các mô hình và các bộ dữ liệu mà **không cần phải huấn luyện lại từ đầu**:

```bash
python -m app.evaluation.report --reports-dir reports
```

### 3.7. Quy Trình Thực Nghiệm Tự Động (Automated Benchmark Pipeline)

Khi bạn muốn cho máy tính tự động chạy huấn luyện tuần tự qua các mô hình và bộ dữ liệu, hệ thống cung cấp quy trình chạy tự động toàn diện. Script sẽ tuần tự chạy từng mô hình trên từng bộ dữ liệu, tự động bắt lỗi an toàn, ghi nhật ký có mốc thời gian vào `reports/benchmark_run_*.log`, **lưu đầy đủ checkpoint, CSV, JSON, Markdown, biểu đồ PNG cho từng bộ dữ liệu** và tự động cập nhật biểu đồ so sánh chéo đa mô hình:

```bash
# Cách 1: Chạy bằng Python (khuyên dùng, hỗ trợ cả Windows, Linux và Kaggle)
# Chạy tất cả các mô hình trên tất cả các bộ dữ liệu:
python -m app.training.run_benchmark --models all --all --epochs 10 --patience 5

# Chạy riêng Mô hình 2 (RoBERTa) trên SROIE và CORD:
python -m app.training.run_benchmark --models roberta --datasets sroie cord --epochs 10

# Cách 2: Chạy qua PowerShell script (Windows)
# Chạy tất cả mô hình trên 3 bộ dữ liệu:
.\scripts\run_benchmark.ps1 -Models "all" -Datasets "sroie", "cord", "docile" -Epochs 10 -Patience 5

# Chạy riêng RoBERTa:
.\scripts\run_benchmark.ps1 -Models "roberta" -Datasets "sroie", "cord" -Epochs 10
```



---

## 4. Ghi Chú về Ứng Dụng Web Minh Họa (Demo Layer — Dành Cho Giai Đoạn Final)

Hệ thống đối soát chứng từ 3 bên (Hóa đơn - Phiếu kho - Đơn đặt hàng) với giao diện web FastAPI + Jinja2 + HTMX là **module ứng dụng minh họa (Demo Layer)**.

> [!NOTE]
> **Vai trò của Web Demo:** Ở giai đoạn nghiên cứu hiện tại, module này được **đóng băng (frozen)** để tập trung tuyệt đối vào việc huấn luyện và so sánh 4 mô hình học sâu. Khi hoàn tất toàn bộ benchmark, mô hình có hiệu năng trích xuất tối ưu nhất sẽ được tích hợp vào ứng dụng này để **trình diễn trực quan (Live Demo) tại buổi báo cáo bảo vệ cuối kỳ (Final Presentation)** trước hội đồng.

Nếu bạn muốn khởi chạy thử nghiệm giao diện web này tại cổng `http://localhost:8000`, có 2 cách thực hiện:

### Cách 1: Khởi chạy nhanh toàn bộ qua Docker Compose (Khuyến nghị cho Final Demo)
```bash
# Tự động dựng cơ sở dữ liệu PostgreSQL, chạy migration và khởi động web server
docker compose -p reconia up -d --build
```
Truy cập tại: `http://localhost:8000` (Dừng: `docker compose -p reconia down`).

### Cách 2: Khởi chạy trực tiếp trên máy chủ cục bộ (Dành cho phát triển & gỡ lỗi)
```bash
# 1. Chỉ bật dịch vụ cơ sở dữ liệu PostgreSQL trong Docker
docker compose -p reconia up -d db

# 2. Chạy migration tạo các bảng dữ liệu qua Alembic
python -m alembic upgrade head

# 3. Khởi động máy chủ web FastAPI ở chế độ reload
python -m uvicorn app.main:app --reload
```
Truy cập tại: `http://localhost:8000`.


---

## 5. Tài Liệu Tham Khảo

Chi tiết về cơ sở lý thuyết, phân tích đối chiếu các bài báo khoa học và kế hoạch thực nghiệm được trình bày tại:
- [docs/SPEC.md](docs/SPEC.md) — Đặc tả hệ thống và khung thực nghiệm thống nhất
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — Tổng quan các công trình nghiên cứu liên quan
- [POLICY.md](POLICY.md) — Quy ước nhóm
