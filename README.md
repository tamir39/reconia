# Reconia: Đánh Giá và So Sánh Các Phương Pháp Học Sâu Cho Trích Xuất Thông Tin Từ Hóa Đơn

*Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction*

**Đồ án:** Dự án CNTT (ĐA161) — HK1/2026-2027  
**Nhóm thực hiện:** Best Stuffs (Phí Vương Tường Tâm - 52300155 · Vũ Hưng Thịnh - 52300159)  
**Giảng viên hướng dẫn:** ThS. Dung Cẩm Quang  

---

## 1. Giới Thiệu & Dữ Liệu Thực Nghiệm

Mục tiêu của đề tài là xây dựng **khung thực nghiệm thống nhất** nhằm huấn luyện, đánh giá và so sánh khách quan sáu mô hình thuộc các nhóm phương pháp khác nhau cho 2 bài toán cốt lõi trong Document AI:
* **KILE (Key Information Localization and Extraction)**: Định vị hộp bao và trích xuất các trường thông tin then chốt (tên công ty, ngày phát hành, tổng tiền, thuế, địa chỉ,...).
* **LIR (Line Item Recognition)**: Nhận dạng cấu trúc bảng và gom nhóm các trường chi tiết (mô tả sản phẩm, số lượng, đơn giá, thành tiền) thành từng dòng hàng sản phẩm hoàn chỉnh (`line_item_id`).

### Sáu mô hình trong phạm vi so sánh:
1. **Đồ thị không gian (Spatial Relational GNN)**: Kết hợp ngữ nghĩa văn bản, 17 đặc trưng hình học nút và vector quan hệ không gian cạnh $r_{ij} \in \mathbb{R}^6$.
2. **Chuỗi tuần tự 1D (Text-only Sequence LM)**: RoBERTa-base bổ sung nhúng tọa độ hộp bao 2D.
3. **Đa phương thức hợp nhất (Multimodal 2D Transformer)**: LayoutLMv3-base kết hợp đồng thời Văn bản, Bố cục 2D và Ảnh thị giác.
4. **Đa phương thức tách rời (Decoupled Layout Transformer)**: LiLT-base tách biệt luồng xử lý văn bản và bố cục.
5. **Qwen2.5-VL**: theo đề xuất của Thịnh; checkpoint 3B-Instruct đề xuất để thử trên Kaggle, chưa hiện thực.
6. **InternVL**: theo đề xuất của Thịnh; checkpoint InternVL2.5-2B đề xuất để thử trên Kaggle, chưa hiện thực.

GNN và RoBERTa đã có các lần chạy được kiểm chứng. LayoutLMv3 đã qua smoke ảnh/OCR cho SROIE/CORD và đang chạy full version 2 trên Kaggle; xem [hướng dẫn](docs/LAYOUTLMV3.md). LiLT và hai VLM chưa hiện thực. LayoutLMv3/DocILE cần bundle ảnh riêng.

Chuẩn bị ảnh/nhãn SROIE và CORD-v2: [hướng dẫn và giới hạn đánh giá](docs/RECEIPT_PREPARATION.md), [notebook GNN receipt](notebooks/kaggle_gnn_receipts.ipynb) mặc định smoke. GNN và RoBERTa đã hoàn tất và kiểm chứng trên cả hai bộ. RoBERTa test entity F1: SROIE 88,37%, CORD 91,99%; xem [bảng so sánh](reports/verified_comparison.md). [Notebook RoBERTa receipt](notebooks/kaggle_roberta_receipts.ipynb) dùng cùng split và mặc định smoke.

Xem [quy trình đánh giá](docs/EVALUATION.md) và [kế hoạch giữa kỳ](docs/MIDTERM.md). Các kết quả trước ngày 24/09 cần tái lập; GNN cũ dùng hàm băm phụ thuộc tiến trình, nên phải huấn luyện lại trước khi sử dụng với bộ mã hóa mới.

Huấn luyện GNN/DocILE trên Kaggle: [notebook](notebooks/kaggle_gnn_docile.ipynb) và [hướng dẫn thiết lập, lưu kết quả, resume](docs/KAGGLE.md).

RoBERTa/DocILE: [notebook huấn luyện đầy đủ](notebooks/kaggle_roberta_docile.ipynb). [Kết quả đã kiểm chứng](reports/verified_comparison.md): GNN test entity F1 66,82%, field-value F1 56,87%; RoBERTa đạt lần lượt 79,69% và 66,39% trên cùng 500 tài liệu test nội bộ. Cả hai lần chạy đã hoàn tất 20 epochs.

### Phân chia dữ liệu thực nghiệm (Train / Val / Test)

Dữ liệu SROIE (`sroie_splits.json`) và DocILE (`docile_splits.json`) được phân chia ngẫu nhiên cố định với `seed=42`, đảm bảo đầy đủ nhãn ground-truth trên cả 3 tập:

| Bộ dữ liệu | Bài báo gốc | Tác vụ | Số nhãn | Train | Val | Test | Test Official |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SROIE** | [Huang et al., ICDAR 2019](https://arxiv.org/abs/2103.10213) | KILE | 4 | 500 | 63 | **63** | — |
| **CORD-v2** | [Park et al., NeurIPS W. 2019](https://openreview.net/forum?id=SJl3z65qhX) | KILE, LIR | 30 | 800 | 100 | **100** | — |
| **DocILE** | [Šimsa et al., ICDAR 2023](https://arxiv.org/abs/2302.05658) | KILE, LIR | 55 | 4.680 | 500 | **500** | 1.000 |

* **Đánh giá thống nhất:** Các mô hình được đo lường trên tập Test độc lập theo chuẩn thực thể nghiêm ngặt (Entity-level Exact Match) với các chỉ số: **`Precision`**, **`Recall`**, **`F1`** và độ trễ suy luận **`Latency (ms)`**.
* Với DocILE, tập `test` (500 mẫu có nhãn) dùng để chấm điểm trực tiếp; tập `test_official` (1.000 mẫu không công bố nhãn của ICDAR 2023) dùng khi xuất file nộp cổng thi [RRC Task 26](https://rrc.cvc.uab.es/?ch=26).

---

## 2. Cấu Trúc Thư Mục Dự Án

```
reconia/
├── app/
│   ├── main.py                # Điểm khởi chạy ứng dụng web FastAPI
│   ├── data/                  # Nạp và chuẩn hóa dữ liệu
│   │   ├── schema.py          # Kiểu dữ liệu trung gian: BBox, Word, Field, Document
│   │   ├── docile.py          # DocileAdapter (train, val, test, test_official)
│   │   ├── cord.py            # CordAdapter (30 thực thể và line_item_id)
│   │   ├── sroie.py           # SroieAdapter (4 trường khóa)
│   │   ├── docile_splits.json # Danh sách 4.680 train / 500 val / 500 test của DocILE
│   │   ├── sroie_splits.json  # Danh sách 500 train / 63 val / 63 test của SROIE
│   │   └── roberta_dataset.py # Chuẩn hóa BBox 2D và căn gióng token cho RoBERTa
│   ├── graph/                 # Dựng đồ thị tài liệu cho mô hình GNN
│   │   ├── builder.py         # Trích xuất 17 đặc trưng hình học nút
│   │   ├── edges.py           # Nối cạnh không gian (same_line, vertical, knn) & vector r_ij
│   │   ├── labels.py          # Gán nhãn BIO và giải mã thực thể
│   │   └── prepare.py         # Tiền xử lý và sinh từ điển nhãn (.cache/)
│   ├── models/                # Mô hình dữ liệu cơ sở dữ liệu SQLAlchemy
│   ├── models_dl/             # Kiến trúc mô hình học sâu
│   │   ├── gnn.py             # Spatial Relational GNN (Node + Edge Conv)
│   │   └── roberta.py         # RoBERTa + 2D Layout Positional Embedding
│   ├── training/              # Huấn luyện mô hình
│   │   ├── train_gnn.py       # Huấn luyện GNN & tự động test
│   │   ├── train_roberta.py   # Huấn luyện RoBERTa & tự động test
│   │   └── run_benchmark.py   # Chạy tự động nhiều mô hình và tập dữ liệu
│   ├── evaluation/            # Đánh giá hiệu năng và xuất báo cáo
│   │   ├── predict.py         # Suy luận, kiểm thử độc lập từ checkpoint
│   │   ├── metrics.py         # Đo đạc Precision, Recall, F1, Latency, Throughput
│   │   └── report.py          # Xuất bảng Markdown, CSV, JSON và đồ thị
│   └── ui/                    # Giao diện web minh họa (FastAPI + Jinja2 + HTMX)
│       ├── routes.py          # Bộ điều hướng hiển thị tài liệu và kết quả
│       └── templates/         # Giao diện HTML trực quan hóa chứng từ
│
├── reports/                   # Kết quả thực nghiệm và báo cáo benchmark
│   ├── benchmark_summary.md   # Bảng tổng hợp so sánh các mô hình
│   ├── figures/               # Biểu đồ huấn luyện và so sánh (*.png)
│   ├── metrics/               # Lịch sử huấn luyện (*_history.csv) và chỉ số chi tiết (*_results.json)
│   ├── predictions/           # File kết quả trích xuất dự đoán
│   └── logs/                  # Nhật ký thực thi huấn luyện và benchmark (*.log)
│
├── checkpoints/               # Trọng số mô hình tối ưu (*_best.pt) và gần nhất (*_latest.pt)
│   └── pretrained/            # Trọng số mô hình nền tải sẵn (roberta-base/)
├── docs/                      # Tài liệu đặc tả kỹ thuật (SPEC.md, RELATED_WORK.md)
└── tests/                     # Bộ kiểm thử tự động (pytest)
```

---

## 3. Bảng Tra Cứu Lệnh Thực Thi

Kích hoạt môi trường trước khi chạy lệnh: `.\.venv\Scripts\activate` (Windows) hoặc `source .venv/bin/activate` (Linux/Kaggle).

### 3.1. Tiền Xử Lý Dữ Liệu & Sinh Từ Điển Nhãn (`app.graph.prepare`)

| Bộ dữ liệu | Lệnh thực thi | Ghi chú |
|---|---|---|
| **Tất cả dữ liệu** | `python -m app.graph.prepare` | Mặc định sinh từ điển và cache cho cả 3 bộ dữ liệu |
| **SROIE** | `python -m app.graph.prepare --dataset sroie` | Chuẩn bị riêng cho SROIE |
| **CORD-v2** | `python -m app.graph.prepare --dataset cord` | Chuẩn bị riêng cho CORD-v2 |
| **DocILE** | `python -m app.graph.prepare --dataset docile` | Chuẩn bị riêng cho DocILE |

### 3.2. Huấn Luyện Mô Hình

*Hệ thống tự động lưu checkpoint tối ưu theo F1 và kiểm thử trên tập test ngay sau khi hoàn thành huấn luyện.*

| Mô hình | Huấn luyện tất cả dữ liệu | Huấn luyện trên bộ dữ liệu chỉ định |
|---|---|---|
| **Spatial Relational GNN** (`gnn`) | `python -m app.training.train_gnn` | `python -m app.training.train_gnn --dataset <sroie\|cord\|docile>` |
| **RoBERTa + 2D Layout** (`roberta`) | `python -m app.training.train_roberta` | `python -m app.training.train_roberta --dataset <sroie\|cord\|docile>` |
| *(Mô hình mở rộng)* | `python -m app.training.train_<model>` | `python -m app.training.train_<model> --dataset <sroie\|cord\|docile>` |

### 3.3. Đánh Giá & Kiểm Thử Độc Lập

*Tự động quét và nạp checkpoint tối ưu tương ứng trong `checkpoints/`. Mặc định đánh giá trên tập `test`.*

| Tác vụ | Lệnh thực thi | Mục tiêu |
|---|---|---|
| **Toàn bộ hệ thống** | `python -m app.evaluation.predict` | Đánh giá tất cả mô hình trên tập test của cả 3 bộ dữ liệu |
| **Theo từng mô hình** | `python -m app.evaluation.predict --model <gnn\|roberta>` | Đánh giá mô hình chỉ định trên cả 3 bộ dữ liệu |
| **Theo mô hình và dữ liệu** | `python -m app.evaluation.predict --model <model> --dataset <sroie\|cord\|docile>` | Đánh giá mô hình trên bộ dữ liệu chỉ định |
| **Tập kiểm thử chính thức DocILE** | `python -m app.evaluation.predict --model <model> --dataset docile --split test_official` | Xuất file dự đoán nộp cổng thi ICDAR RRC (1.000 mẫu) |

### 3.4. Chạy Tự Động Bộ Thí Nghiệm Huấn Luyện

| Mục đích | Lệnh thực thi | Hành động |
|---|---|---|
| **Chạy toàn bộ** | `python -m app.training.run_benchmark` | Huấn luyện và đánh giá toàn bộ mô hình trên cả 3 bộ dữ liệu |
| **Thử nghiệm nhanh (Dry run)** | `python -m app.training.run_benchmark --limit-train 50 --limit-val 20` | Giới hạn số mẫu để kiểm tra nhanh luồng thực thi |
| **Chạy riêng mô hình** | `python -m app.training.run_benchmark --models roberta` | Huấn luyện mô hình chỉ định trên cả 3 bộ dữ liệu |
| **Chạy theo dữ liệu** | `python -m app.training.run_benchmark --datasets sroie cord` | Huấn luyện trên các bộ dữ liệu chỉ định |

### 3.5. Tổng Hợp Báo Cáo & Biểu Đồ So Sánh Hiệu Năng (`app.evaluation.report`)

*Quét toàn bộ kết quả trong `reports/metrics/*_results.json` để tổng hợp bảng so sánh benchmark và vẽ biểu đồ so sánh hiệu năng các mô hình (`benchmark_comparison.png`).*

| Tác vụ | Lệnh thực thi | Mục tiêu |
|---|---|---|
| **Tổng hợp toàn diện** | `python -m app.evaluation.report` | Xuất `benchmark_summary.md` và vẽ biểu đồ so sánh hiệu năng `benchmark_comparison.png` |
| **Chỉ định thư mục** | `python -m app.evaluation.report --report-dir <path>` | Chỉ định thư mục chứa báo cáo (mặc định: `reports/`) |

---

## 4. Bảng Tham Số Dòng Lệnh Cốt Lõi

### 4.1. Tham Số Tiền Xử Lý Dữ Liệu (`app.graph.prepare`)

| Tham số | Mặc định | Ý nghĩa & Tùy chọn |
|---|:---:|---|
| `--dataset` | `all` | Bộ dữ liệu xử lý: `all`, `sroie`, `cord`, `docile` |
| `--splits` | `all` | Tập dữ liệu: `all`, `train`, `val`, `test` |
| `--limit` | `None` | Giới hạn số tài liệu xử lý trên mỗi split |
| `--cache-dir` | `.cache/` | Thư mục lưu trữ cache đồ thị `.npz` và từ điển nhãn BIO `.json` |
| `--root` | `None` | Đường dẫn thư mục dữ liệu gốc (mặc định: tự động định vị qua `app.data.paths`) |
| `--rebuild` | `False` | Bỏ qua cache hiện có và dựng lại toàn bộ đồ thị |
| `--k` | `6` | Số láng giềng k-NN cho cạnh không gian đồ thị |
| `--no-knn` | `False` | Tắt sinh cạnh k-láng giềng gần nhất |
| `--no-same-line` | `False` | Tắt sinh cạnh nối các từ trên cùng dòng |
| `--no-vertical` | `False` | Tắt sinh cạnh nối các từ thẳng hàng dọc |

### 4.2. Tham Số Huấn Luyện (`app.training.train_gnn` & `app.training.train_roberta`)

| Nhóm | Tham số | Mặc định GNN | Mặc định RoBERTa | Ý nghĩa & Tùy chọn |
|---|---|:---:|:---:|---|
| **Chung** | `--dataset` | `all` | `all` | Bộ dữ liệu: `all`, `sroie`, `cord`, `docile` |
| | `--epochs` | `20` | `20` | Số epoch tối đa trong một đợt huấn luyện |
| | `--patience` | `5` | `5` | Số epoch chờ trước khi dừng sớm (`0` để tắt) |
| | `--monitor-metric`| `f1` | `f1` | Chỉ số giám sát chọn checkpoint tối ưu: `f1` hoặc `val_loss` |
| | `--device` | `auto` | `auto` | Thiết bị tính toán: `cuda`, `cpu`, hoặc `auto` |
| | `--resume` | `None` | `None` | Đường dẫn checkpoint `.pt` để tiếp tục huấn luyện |
| | `--weighted-loss`   | `True` | `True` | Trọng số phạt nhãn hiếm trong CrossEntropyLoss (`True`: phạt nặng nhãn hiếm; `False`: CrossEntropy chuẩn) |
| | `--save-dir` | `checkpoints/` | `checkpoints/` | Thư mục lưu checkpoint mô hình |
| | `--report-dir` | `reports/` | `reports/` | Thư mục lưu nhật ký và báo cáo thực nghiệm |
| | `--cache-dir` | `.cache/` | `.cache/` | Thư mục lưu cache đồ thị và từ điển nhãn |
| | `--limit-train` | `None` | `None` | Giới hạn số mẫu huấn luyện khi chạy thử nghiệm nhanh |
| | `--limit-val` | `None` | `None` | Giới hạn số mẫu đánh giá khi chạy thử nghiệm nhanh |
| **Tối ưu** | `--lr` | `1e-3` | `5e-5` | Tốc độ học của mô hình nền (Backbone LR) |
| | `--head-lr` | — | `2e-4` | Tốc độ học của tầng phân loại (Head LR) |
| | `--weight-decay` | `1e-4` | `0.01` | Hệ số suy giảm trọng số (Weight decay) |
| | `--batch-size` | `1` | `4` | Kích thước batch forward |
| | `--batch-accum` | `16` | `4` | Số bước tích lũy gradient |
| | `--clip-grad-norm`| `1.0` | `1.0` | Ngưỡng cắt tỉa gradient (`<= 0` để tắt) |
| **Kiến trúc GNN** | `--hidden-dim` | `128` | — | Số chiều biểu diễn ẩn tại mỗi nút từ |
| | `--num-layers` | `2` | — | Số tầng tích chập không gian (Spatial Relational Edge Conv) |
| | `--dropout` | `0.2` | — | Tỷ lệ dropout |

### 4.3. Tham Số Đánh Giá & Suy Luận (`app.evaluation.predict`)

| Tham số | Mặc định | Ý nghĩa & Tùy chọn |
|---|:---:|---|
| `--model` | `auto` | Kiến trúc mô hình: `auto`, `gnn`, `roberta` |
| `--dataset` | `all` | Bộ dữ liệu: `all`, `sroie`, `cord`, `docile` |
| `--checkpoint` | `None` | Đường dẫn checkpoint `.pt` (mặc định: tự tìm file `_best.pt` tương ứng) |
| `--split` | `test` | Tập dữ liệu đánh giá: `test`, `val`, `train`, `test_official` |
| `--output` | `None` | Đường dẫn file JSON lưu kết quả dự đoán (mặc định: `reports/predictions/`) |
| `--limit` | `None` | Giới hạn số tài liệu suy luận |
| `--all` | `False` | Quét và đánh giá toàn bộ mô hình trên cả 3 bộ dữ liệu |
| `--cache-dir` | `None` | Thư mục cache từ điển nhãn (mặc định: `.cache/`) |
| `--device` | `auto` | Thiết bị tính toán: `cuda`, `cpu`, hoặc `auto` |

### 4.4. Tham Số Bộ Thí Nghiệm Benchmark (`app.training.run_benchmark`)

| Tham số | Mặc định | Ý nghĩa & Tùy chọn |
|---|:---:|---|
| `--models` | `all` | Danh sách mô hình chạy: `all`, `gnn`, `roberta` |
| `--datasets` | `all` | Danh sách bộ dữ liệu: `all`, `sroie`, `cord`, `docile` |
| `--epochs` | `20` | Số epoch tối đa cho mỗi cặp mô hình và dữ liệu |
| `--patience` | `5` | Ngưỡng dừng sớm early stopping (`0` để tắt) |
| `--lr` | `None` | Tốc độ học (mặc định theo từng mô hình) |
| `--batch-accum` | `None` | Số bước tích lũy gradient |
| `--monitor-metric`| `f1` | Chỉ số giám sát lưu checkpoint: `f1` hoặc `val_loss` |
| `--limit-train` | `None` | Giới hạn mẫu huấn luyện khi chạy thử nhanh |
| `--limit-val` | `None` | Giới hạn mẫu đánh giá khi chạy thử nhanh |
| `--save-dir` | `None` | Thư mục lưu checkpoint (mặc định: `checkpoints/`) |
| `--report-dir` | `None` | Thư mục lưu báo cáo (mặc định: `reports/`) |
| `--device` | `auto` | Thiết bị tính toán: `cuda`, `cpu`, hoặc `auto` |

### 4.5. Tham Số Tổng Hợp Báo Cáo (`app.evaluation.report`)

| Tham số | Mặc định | Ý nghĩa & Tùy chọn |
|---|:---:|---|
| `--report-dir` | `reports/` | Đường dẫn đến thư mục chứa kết quả và báo cáo |

---

## 5. Báo Cáo Kết Quả Thực Nghiệm

Kết quả thực nghiệm được tự động xuất vào thư mục `reports/`:
* `reports/benchmark_summary.md`: Bảng tổng hợp so sánh hiệu năng các mô hình trên tập Test.
* `reports/figures/benchmark_comparison.png`: Biểu đồ so sánh trực quan F1 và Latency.
* `reports/figures/<model>_<dataset>_training_curves.png`: Đồ thị Loss và F1 qua các epoch.
* `reports/<model>_<dataset>_report.md`: Báo cáo chi tiết theo từng loại thực thể.
* `reports/metrics/`: Dữ liệu lịch sử huấn luyện dạng CSV và kết quả chi tiết dạng JSON.
* `reports/predictions/`: Kết quả trích xuất dự đoán dạng JSON.
* `reports/logs/`: Nhật ký thực thi quá trình huấn luyện và benchmark.

---

## 6. Chạy Kiểm Thử Tự Động

```bash
pytest tests/ -v
```

---

## 7. Ứng Dụng Web Minh Họa

Dự án cung cấp giao diện web minh họa trực quan (xây dựng bằng **FastAPI**, **Jinja2**, **HTMX** và **TailwindCSS**) phục vụ công tác demo nghiệm thu, trực quan hóa và đối chiếu kết quả giữa các mô hình học sâu:
* **Trực quan hóa tài liệu:** Hiển thị trực quan hóa đơn, biên lai cùng hệ thống hộp bao (Bounding Box) từ OCR.
* **Đối chiếu thực thể:** Kiểm tra và so sánh trực quan các trường thông tin trích xuất giữa nhãn gốc (Ground Truth) và kết quả dự đoán của các mô hình (GNN vs. Baseline Transformer).
* **Quản lý danh sách chứng từ:** Hỗ trợ lọc theo loại chứng từ, tìm kiếm theo số hóa đơn và tra cứu chi tiết từng trang tài liệu.

### 7.1. Chạy Cục Bộ

```bash
# 1. Khởi tạo và cập nhật cấu trúc cơ sở dữ liệu
python -m alembic upgrade head

# 2. Khởi chạy máy chủ Web FastAPI với chế độ tải lại tự động (reload)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Truy cập giao diện web tại: `http://localhost:8000`

### 7.2. Chạy Qua Docker

```bash
docker compose -p reconia up -d --build
```
Kiểm tra tình trạng hệ thống:
* **Web UI:** `http://localhost:8000`
* **Kiểm tra tình trạng (Liveness):** `http://localhost:8000/health`
* **Kiểm tra kết nối CSDL (Readiness):** `http://localhost:8000/health/db`

---

## 8. Tài Liệu Tham Khảo

- [docs/SPEC.md](docs/SPEC.md) — Đặc tả hệ thống, phương pháp và khung thực nghiệm thống nhất
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — Tổng quan các công trình nghiên cứu liên quan
- [POLICY.md](POLICY.md) — Quy ước nhóm



