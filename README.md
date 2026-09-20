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
1. **Mô hình Đồ thị (Graph-based)**: Mạng tích chập đồ thị quan hệ không gian (*Spatial Relational GCN*) tự hiện thực bằng PyTorch thuần, kết hợp Text Semantics + 17 đặc trưng hình học và vector quan hệ không gian cạnh (dựa trên [Liu et al., NAACL 2019](https://aclanthology.org/N19-1138/) và [PICK, ICPR 2020](https://arxiv.org/abs/2004.07464)). Huấn luyện trên GPU cá nhân (GTX 1650 Ti 4GB).
2. **Mô hình Ngôn ngữ tuần tự 1D (Text-only Sequence LM)**: RoBERTa token classification (chuẩn baseline của [DocILE Benchmark, ICDAR 2023](https://arxiv.org/abs/2302.05658)).
3. **Mô hình Đa phương thức 2D (Multimodal 2D Transformer)**: LayoutLMv3 kết hợp đồng thời Text Tokens + 2D Bounding Boxes + Image Patches ([Huang et al., ACM MM 2022](https://arxiv.org/abs/2204.08387)).
4. **Mô hình Bố cục tách rời (Decoupled Layout Transformer)**: LiLT tách riêng nhánh ngôn ngữ và nhánh bố cục 2D độc lập ([Wang et al., ACL 2022](https://arxiv.org/abs/2202.13669)).

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
│   │   ├── roberta_dataset.py# PyTorch Dataset chuẩn hóa BBox 2D và căn gióng nhãn sub-word cho RoBERTa
│   │   ├── paths.py          # Cơ chế tự động dò tìm đường dẫn dữ liệu di động
│   │   └── sroie_splits.json # Phân chia ngẫu nhiên 500/63/63 tái lập được của SROIE
│   │
│   ├── graph/                # Tầng xây dựng đồ thị tài liệu (Document Graph cho GNN)
│   │   ├── builder.py        # Dựng DocumentGraph, trích xuất 17 đặc trưng hình học nút
│   │   ├── edges.py          # Nối cạnh không gian (kNN theo trang, cùng dòng, cùng cột) & vector r_ij
│   │   ├── labels.py         # Gán nhãn BIO, tính thống kê độ phủ (ceiling) và giải mã thực thể
│   │   ├── cache.py          # Bộ đệm lưu đồ thị (.npz) và từ điển nhãn (.json)
│   │   └── prepare.py        # CLI tiền xử lý và sinh cache đồ thị cho các bộ dữ liệu
│   │
│   ├── models_dl/            # Các kiến trúc mô hình học sâu
│   │   ├── gnn.py            # Mô hình 1: Spatial Relational GNN (Node + Edge Conv)
│   │   └── roberta.py        # Mô hình 2: RoBERTa + 2D Layout Positional Embedding
│   │
│   ├── training/             # Vòng lặp huấn luyện, checkpointing, profiling tài nguyên
│   │   ├── train_gnn.py      # Pipeline huấn luyện GNN trên SROIE, CORD, DocILE
│   │   ├── train_roberta.py  # Pipeline huấn luyện RoBERTa trên SROIE, CORD, DocILE
│   │   └── run_benchmark.py  # Bộ điều phối tự động chạy suite so sánh đối đầu đa mô hình
│   │
│   ├── evaluation/           # Đo đạc hiệu năng: Precision, Recall, F1, Latency & VRAM
│   │   ├── metrics.py        # Tính Precision, Recall, F1 và profiler tài nguyên phần cứng
│   │   └── report.py         # Xuất báo cáo Markdown, CSV, JSON và vẽ biểu đồ so sánh
│   │
│   ├── models/               # [Đóng băng] Lược đồ CSDL PostgreSQL cho web demo
│   │   └── entities.py       # SQLAlchemy ORM models (Invoice, LineItem, PO, GoodsReceipt)
│   ├── ui/                   # [Đóng băng] Giao diện web đối soát demo (Jinja2 + HTMX)
│   └── main.py               # [Đóng băng] FastAPI app cho web demo
│
├── reports/                  # Kết quả thực nghiệm và biểu đồ
│   ├── figures/              # Biểu đồ huấn luyện và so sánh mô hình
│   ├── benchmark_summary.md  # Bảng tổng hợp so sánh giữa các mô hình trên các tập dữ liệu
│   ├── benchmark_run_*.log   # Log thực thi benchmark suite
│   ├── *_history.csv         # Lịch sử chi tiết từng epoch (loss, metrics, latency, vram)
│   ├── *_results.json        # Dữ liệu JSON phục vụ tổng hợp kết quả
│   └── *_report.md           # Báo cáo chi tiết từng mô hình
├── docs/                     # Tài liệu thiết kế và nghiên cứu học thuật
│   ├── SPEC.md               # Đặc tả bài toán, phương pháp, dữ liệu và khung thực nghiệm
│   ├── RELATED_WORK.md       # Tổng quan các công trình nghiên cứu liên quan
│   ├── AGENT.md              # [Đóng băng] Ghi chú thiết kế Agent kiểm chứng cho web demo
│   └── ERD.md                # [Đóng băng] Lược đồ CSDL quan hệ cho web demo
│
├── tests/                    # Bộ kiểm thử tự động
├── requirements.txt          # Danh sách thư viện phụ thuộc của dự án
├── .env.example              # Mẫu biến môi trường cấu hình đường dẫn và CSDL
└── POLICY.md                 # Quy ước đóng góp git, đặt tên nhánh và commit
```

---

## 3. Cài Đặt & Hướng Dẫn Sử Dụng

### 3.1. Thiết lập Môi trường Ảo (Virtualenv)

Khuyến nghị sử dụng **Python 3.10**:

```bash
# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo:
# Trên Windows (PowerShell / CMD):
.\.venv\Scripts\activate
# Trên Linux / macOS / Kaggle:
source .venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc:
pip install -r requirements.txt

# 4. (Khuyến nghị) Cài đặt PyTorch có hỗ trợ GPU NVIDIA CUDA (cho GTX 1650 Ti):
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
# Với bộ dữ liệu SROIE (500 train, 63 val, 63 test):
python -m app.graph.prepare --dataset sroie --splits train val test

# Với bộ dữ liệu CORD-v2 (800 train, 100 validation, 100 test):
python -m app.graph.prepare --dataset cord --splits train validation test

# Với bộ dữ liệu DocILE (5.180 train, 500 val):
python -m app.graph.prepare --dataset docile --splits train val
```

### 3.5. Huấn Luyện Mô Hình Học Sâu (Training Pipelines)

Hệ thống cung cấp quy trình huấn luyện độc lập cho từng kiến trúc mô hình với các cơ chế tự động:
* **Xử lý mất cân bằng lớp:** Tự động tính toán và bù trọng số lớp (`compute_class_weights`) trong hàm mất mát.
* **Theo dõi hiệu năng phần cứng:** Giám sát mức chiếm dụng VRAM đỉnh (`Peak GPU VRAM`) và độ trễ suy luận trung bình mỗi tài liệu (`Latency ms/doc`).
* **Quản lý Checkpoint kép:** Tự động lưu mô hình tốt nhất (`*_best.pt`) theo chỉ số giám sát và mô hình gần nhất (`*_latest.pt`) để sẵn sàng phục hồi.
* **Theo dõi tiến trình:** Bảng kết quả từng epoch hiển thị trực tiếp trên Terminal:
  ```text
  ===================================================================================
  Epoch  | TrainLoss  | ValLoss    | Precision  | Recall     | F1         | Latency(ms)
  ===================================================================================
  ```
  * `TrainLoss` / `ValLoss`: CrossEntropy loss trên tập train và validation.
  * `Precision` / `Recall` / `F1`: Độ chính xác, độ bao phủ và F1 trích xuất thực thể (Entity-level Exact Match).
  * `Latency (ms)`: Thời gian suy luận trung bình mỗi tài liệu (ms).

#### Các lệnh huấn luyện tiêu biểu:

**1. Huấn luyện Mô hình 1: Spatial Relational GNN**
```bash
# Huấn luyện trên SROIE (Mặc định: 20 epoch, Effective Batch Size = 16, LR = 1e-3, theo dõi F1)
python -m app.training.train_gnn --dataset sroie

# Huấn luyện trên CORD-v2 theo dõi Entity F1
python -m app.training.train_gnn --dataset cord --monitor-metric f1

# Huấn luyện trên DocILE (tập con 500 train, 100 val)
python -m app.training.train_gnn --dataset docile --limit-train 500 --limit-val 100
```

**2. Huấn luyện Mô hình 2: RoBERTa + 2D Layout Positional Embedding**
```bash
# Huấn luyện trên SROIE (Mặc định: 20 epoch, batch size 4 x accum 4 = 16, LR = 3e-5)
python -m app.training.train_roberta --dataset sroie

# Huấn luyện trên CORD-v2 theo dõi Entity F1
python -m app.training.train_roberta --dataset cord --monitor-metric f1

# Huấn luyện trên DocILE (tập con 500 train, 100 val)
python -m app.training.train_roberta --dataset docile --limit-train 500 --limit-val 100
```

**3. Tiếp tục huấn luyện từ Checkpoint (Resume)**
```bash
# Nạp lại toàn bộ trọng số và trạng thái optimizer để tiếp tục học
python -m app.training.train_gnn --dataset sroie --resume checkpoints/gnn_sroie_latest.pt --epochs 30
python -m app.training.train_roberta --dataset sroie --resume checkpoints/roberta_sroie_latest.pt --epochs 30
```

**4. Chạy kiểm thử nhanh (Fast Debugging)**
```bash
# Chỉ nạp 30 tài liệu train và 10 tài liệu val để kiểm tra mã nguồn trong 1 phút
python -m app.training.train_gnn --dataset sroie --limit-train 30 --limit-val 10 --epochs 3
python -m app.training.train_roberta --dataset sroie --limit-train 30 --limit-val 10 --epochs 3
```

---

### 3.6. Cẩm Nang Tra Cứu Tham Số Dòng Lệnh (CLI Reference)

Mọi kịch bản huấn luyện đều tuân thủ quy ước tham số thống nhất:

#### Bảng 1: Nhóm tham số Điều Khiển Chung (Áp dụng cho cả `train_gnn` và `train_roberta`)

| Tham số | Kiểu | Mặc định | Ý nghĩa & Các lựa chọn khả dụng |
| :--- | :--- | :--- | :--- |
| **`--dataset`** | chuỗi | `sroie` | Bộ dữ liệu mục tiêu: `sroie`, `cord`, `docile` |
| **`--epochs`** | int | `20` | Số lượng epoch tối đa trong một đợt huấn luyện |
| **`--patience`** | int | `5` | Số epoch dừng sớm nếu chỉ số theo dõi không tăng (`0` để tắt) |
| **`--monitor-metric`** | chuỗi | `f1` | Chỉ số giám sát để lưu model tốt nhất và dừng sớm:<br>• `f1` *(mặc định: F1 Exact Match theo chuẩn quốc tế)*<br>• `val_loss` *(Validation Loss)* |
| **`--device`** | chuỗi | `auto` | Thiết bị tính toán: `cuda`, `cpu`, hoặc `auto` |
| **`--resume`** | path | `None` | Đường dẫn tệp checkpoint `.pt` để khôi phục trạng thái huấn luyện |
| **`--seed`** | int | `42` | Hạt giống ngẫu nhiên đảm bảo tính tái lập khoa học |
| **`--limit-train`** | int | `None` | Giới hạn số tài liệu huấn luyện (dùng khi thử nghiệm nhanh) |
| **`--limit-val`** | int | `None` | Giới hạn số tài liệu kiểm định (dùng khi thử nghiệm nhanh) |
| **`--save-dir`** | path | `checkpoints/`| Thư mục lưu trữ trọng số mô hình |
| **`--report-dir`** | path | `reports/` | Thư mục lưu trữ báo cáo, bảng tính và biểu đồ |
| **`--no-weighted-loss`**| flag | *Tắt* | Tắt cơ chế cân bằng trọng số lớp trong Loss |

#### Bảng 2: Nhóm tham số Tối Ưu Hóa & Cấu Hình Batch (Đảm bảo công bằng phần cứng)

| Tham số | Mặc định GNN | Mặc định RoBERTa | Ý nghĩa khoa học |
| :--- | :--- | :--- | :--- |
| **`--lr`** | `1e-3` | `3e-5` | Tốc độ học AdamW (GNN học từ đầu cần LR lớn; Transformer fine-tuning cần LR nhỏ). |
| **`--batch-size`** | *(1 doc)* | `4` | Số tài liệu nạp vào GPU cùng lúc (RoBERTa đặt 4 để giữ VRAM $\approx 1.6\text{ GB}$). |
| **`--batch-accum`** | `16` | `4` | Số bước tích lũy đạo hàm trước khi cập nhật trọng số.<br>$\Rightarrow$ **Effective Batch Size = 16** cho cả 2 mô hình ($1 \times 16 = 16$ và $4 \times 4 = 16$). |
| **`--weight-decay`**| `1e-4` | `0.01` | Hệ số suy giảm trọng số L2 (RoBERTa dùng decoupled decay chuẩn). |
| **`--clip-grad-norm`**| `1.0` | `1.0` | Ngưỡng cắt tỉa chuẩn gradient chống bùng nổ đạo hàm. |

#### Bảng 3: Nhóm tham số Kiến Trúc Đặc Thù (Chỉ dành cho `train_gnn`)

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **`--hidden-dim`** | int | `128` | Số chiều vector biểu diễn ẩn tại mỗi nút từ |
| **`--num-layers`** | int | `2` | Số tầng tích chập không gian (Spatial Relational Edge Conv) |
| **`--dropout`** | float | `0.2` | Tỷ lệ dropout chống quá khớp |

#### Bảng 4: Nhóm tham số của Bộ Điều Phối Benchmark Tự Động (`run_benchmark`)

| Tham số | Kiểu | Mặc định | Ý nghĩa & Lựa chọn |
| :--- | :--- | :--- | :--- |
| **`--models`** | list | `all` | Danh sách mô hình cần chạy: `gnn`, `roberta`, hoặc `all` |
| **`--datasets`** | list | `sroie cord docile`| Danh sách tập dữ liệu: `sroie`, `cord`, `docile` |
| **`--all`** | flag | *Bật* | Chạy toàn diện cả 3 tập dữ liệu (`sroie cord docile`) |
| **`--epochs`** | int | `20` | Số epoch áp dụng đồng loạt cho mọi thí nghiệm trong suite |
| **`--patience`** | int | `5` | Ngưỡng dừng sớm cho từng mô hình |
| **`--monitor-metric`** | chuỗi | `f1` | Metric theo dõi truyền xuống các mô hình (`f1`, `val_loss`) |
| **`--docile-limit-train`**| int | `500` | Giới hạn train cho DocILE (để `0` nếu muốn train toàn bộ 5.180 tài liệu) |
| **`--docile-limit-val`** | int | `100` | Giới hạn val cho DocILE (để `0` nếu muốn val toàn bộ 500 tài liệu) |

---

### 3.7. Quy Trình Thực Nghiệm Tự Động Toàn Diện (Automated Benchmark Suite)

Khi muốn chạy so sánh đối đầu tự động đa mô hình qua đêm (tự động chạy tất cả mô hình trên cả 3 bộ dữ liệu, lưu checkpoint và vẽ bảng so sánh):

```bash
# Chạy FULL tự động toàn bộ (Tất cả mô hình x Cả 3 bộ dữ liệu x 20 epoch):
python -m app.training.run_benchmark

# (Tùy chọn) Chạy riêng 1 mô hình trên cả 3 bộ dữ liệu:
python -m app.training.run_benchmark --models roberta

# (Tùy chọn) Chạy trên 2 bộ dữ liệu nhỏ (SROIE và CORD):
python -m app.training.run_benchmark --datasets sroie cord
```

---

### 3.8. Báo Cáo Thực Nghiệm & Đồ Thị (Reports)

Sau khi huấn luyện hoặc chạy benchmark, kết quả được lưu tại thư mục `reports/`:
* **`reports/<model>_<dataset>_report.md`**: Báo cáo Markdown chi tiết (Precision, Recall, F1, Latency, VRAM, kết quả theo từng loại nhãn và lịch sử huấn luyện).
* **`reports/<model>_<dataset>_history.csv`**: Dữ liệu lịch sử huấn luyện dạng bảng (`epoch, train_loss, val_loss, precision, recall, f1, latency_ms, peak_vram_mb`).
* **`reports/<model>_<dataset>_results.json`**: File JSON chứa toàn bộ metric và cấu hình phục vụ phân tích.
* **`reports/figures/<model>_<dataset>_training_curves.png`**: Đồ thị 2 ô biểu diễn:
  * Hội tụ Loss (Train vs Val)
  * Chất lượng trích xuất (Precision, Recall và F1)
* **`reports/benchmark_summary.md` & `reports/figures/benchmark_comparison.png`**: Bảng tổng hợp đối đầu giữa các mô hình và biểu đồ 3 ô so sánh F1 (%), Latency (ms), VRAM (MB).

Để tạo lại báo cáo hoặc cập nhật bảng tổng hợp từ các file kết quả đã có:
```bash
python -m app.evaluation.report --reports-dir reports
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
