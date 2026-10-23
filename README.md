# Reconia

- Tiếng Việt: Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin từ hóa đơn
- Tiếng Anh: Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction

## 1. Phạm vi nghiên cứu

- **3 tập dữ liệu:** `sroie`, `cord_v2`, `wildreceipt`.
- **6 mô hình:**
  1. `roberta`: Mô hình thuần văn bản (Text-only).
  2. `bros`: Mô hình kết hợp văn bản và tọa độ 2D (Text + Layout).
  3. `pick`: Mạng đồ thị kết hợp văn bản, tọa độ 2D và hình ảnh (Graph-based Multimodal).
  4. `layoutlmv3`: Mô hình đa phương thức kết hợp văn bản, tọa độ 2D và hình ảnh (Multimodal Transformer).
  5. `donut`: Mô hình trực tiếp từ ảnh sang văn bản cấu trúc (OCR-free).
  6. `qwen_vl`: Mô hình thị giác - ngôn ngữ lớn (Vision-Language Model - VLM).

## 2. Cấu trúc thư mục

```
reconia/
├── .cache/
│   └── <dataset>/
│       ├── test.jsonl
│       ├── train.jsonl
│       └── val.jsonl
├── checkpoints/
│   └── <dataset>/
│       ├── labels.json
│       └── <model>/
│           ├── best_weights.pt
│           └── last_checkpoint.pt
├── models/
│   └── <model>/
│       ├── dataset.py
│       ├── model.py
│       └── train.py
├── notebooks/
│   └── <model>_model.ipynb
├── prepare/
│   ├── prepare_cord_v2.py
│   ├── prepare_sroie.py
│   └── prepare_wildreceipt.py
├── results/
│   └── <dataset>/
│       ├── <model>/
│       │   ├── history.csv
│       │   ├── loss.png
│       │   ├── metrics.png
│       │   ├── test_metrics.json
│       │   └── train.log
│       ├── comparison.md
│       └── comparison.png
├── tools/
│   ├── compare.py
│   └── download_weights.py
├── utils/
│   ├── common.py
│   ├── metrics.py
│   └── plots.py
├── weights/
│   └── <model>/
├── .gitignore
├── README.md
└── requirements.txt
```

## 3. Cài đặt

```bash
git clone -b sub-dev-02 https://github.com/tamir39/reconia.git
cd reconia
pip install -r requirements.txt
```

## 4. Dữ liệu gốc

Dữ liệu được lưu trữ trên Kaggle tại [`vhngthnh/reconia-data`](https://www.kaggle.com/datasets/vhngthnh/reconia-data):
- **SROIE:** [ICDAR 2019 Robust Reading Challenge](https://rrc.cvc.uab.es/?ch=13).
- **CORD-v2:** [clovaai/cord](https://github.com/clovaai/cord).
- **WildReceipt:** [MMOCR KIE Dataset](https://github.com/open-mmlab/mmocr/blob/12558969ee6f11b469a8e964a57730d5df3df6b1/docs/en/datasets/kie.md).

## 5. Quy trình thực nghiệm

### Bước 1: Chuẩn bị dữ liệu
Chuyển đổi dữ liệu gốc sang JSONL trong `.cache/<dataset>/` và xuất danh sách nhãn ra `checkpoints/<dataset>/labels.json`:

```bash
python -m prepare.prepare_sroie
python -m prepare.prepare_cord_v2
python -m prepare.prepare_wildreceipt
```

### Bước 2: Tải trọng số mô hình
Tải trọng số tiền huấn luyện từ Hugging Face về `weights/<model>/`:

```bash
python -m tools.download_weights --model <model>
```

| Tham số | Mặc định | Kiểu / Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--model` | `"all"` | `roberta`, `bros`, `layoutlmv3`, `donut`, `qwen_vl`, `all` | Tên mô hình cần tải (hoặc `"all"` để tải toàn bộ) |

### Bước 3: Huấn luyện và đánh giá

Thiết lập chung:
- Batch size hiệu dụng: 8 (`batch_size * accum_steps = 8`)
- Optimizer: AdamW (`weight_decay=0.01`, không áp dụng cho `bias` và các lớp chuẩn hóa)
- Random seed: 42 (`seed=42`)
- Số epoch tối đa: 30 (`epochs=30`)
- Số epoch chạy tối thiểu: 5 (`min_epochs=5`)
- Dừng sớm: 5 epoch không cải thiện (`patience=5`)

Lệnh thực thi:

```bash
python -m models.<model>.train --dataset <dataset>
```

#### Cấu hình mặc định theo mô hình:

| Mô hình | Learning Rate | `batch_size` | `accum_steps` |
| :--- | :--- | :---: | :---: |
| `roberta` | `lr = 3e-5` | `8` | `1` |
| `bros` | `lr = 3e-5` | `4` | `2` |
| `pick` | `lr_backbone = 1e-4`, `lr_scratch = 1e-3` | `4` | `2` |
| `layoutlmv3` | `lr = 3e-5` | `4` | `2` |
| `donut` | `lr = 3e-5` | `2` | `4` |
| `qwen_vl` | `lr = 1e-4` | `1` | `8` |

#### Bảng tham số dòng lệnh:

| Tham số | Mặc định | Kiểu / Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--dataset` | `"all"` | `sroie`, `cord_v2`, `wildreceipt`, `all` | Tập dữ liệu huấn luyện |
| `--epochs` | `30` | `int` | Số epoch tối đa |
| `--batch_size` | Theo mô hình | `int` | Batch size mỗi step |
| `--accum_steps` | Theo mô hình | `int` | Số bước tích lũy gradient |
| `--lr` | Theo mô hình | `float` | Learning rate cho các mô hình đơn LR (`roberta`, `bros`, `layoutlmv3`, `donut`, `qwen_vl`) |
| `--lr_backbone` | `1e-4` | `float` | Learning rate cho pretrained ResNet-50 backbone (chỉ dùng cho `pick`) |
| `--lr_scratch` | `1e-3` | `float` | Learning rate cho các tầng huấn luyện từ đầu: BiLSTM, GCN, CRF (chỉ dùng cho `pick`) |
| `--min_epochs` | `5` | `int` | Số epoch chạy tối thiểu trước khi xét dừng sớm |
| `--patience` | `5` | `int` | Số epoch chờ không cải thiện trước khi dừng sớm |
| `--resume` | `False` | `flag` | Tiếp tục huấn luyện từ checkpoint gần nhất (`last_checkpoint.pt`) |

Kết quả lưu tại `results/<dataset>/<model>/`.

### Bước 4: So sánh kết quả
Tổng hợp kết quả các mô hình tại `results/<dataset>/`:

```bash
python -m tools.compare
```

Lệnh tự động trích xuất bảng `comparison.md` và biểu đồ `comparison.png` theo field-level F1 cho từng tập dữ liệu.

### Thực nghiệm trên Jupyter Notebook
Mỗi mô hình có notebook tương ứng tại `notebooks/<model>_model.ipynb`, hỗ trợ thực thi trên máy cục bộ và Kaggle.