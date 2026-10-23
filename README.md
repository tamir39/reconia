# Reconia

- Tiếng Việt: Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin từ hóa đơn
- Tiếng Anh: Evaluation and Comparison of Deep Learning Methods for Invoice Information Extraction

## 1. Phạm vi nghiên cứu

- **3 tập dữ liệu:** `sroie`, `cord_v2`, `wildreceipt`.
- **6 mô hình:**
  1. `roberta`: Xử lý chuỗi văn bản thuần túy.
  2. `bros`: Kết hợp văn bản và tọa độ 2D.
  3. `pick`: Mạng đồ thị kết hợp văn bản, tọa độ 2D và hình ảnh.
  4. `layoutlmv3`: Đa phương thức kết hợp văn bản, tọa độ 2D và hình ảnh.
  5. `donut`: Biến đổi trực tiếp từ ảnh sang văn bản cấu trúc (OCR-free).
  6. `qwen_vl`: Thị giác - ngôn ngữ lớn (VLM) theo câu lệnh chỉ dẫn (Qwen2.5-VL-3B-Instruct).

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

| Tham số | Mặc định | Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--model` | `"all"` | `roberta`, `bros`, `layoutlmv3`, `donut`, `qwen_vl`, `all` | Tên mô hình cần tải hoặc toàn bộ |

### Bước 3: Huấn luyện và đánh giá

Thiết lập chung:
- Kích thước batch hiệu dụng: 8 (`batch_size * accum_steps = 8`)
- Bộ tối ưu hóa: AdamW (`weight_decay=0.01`, loại trừ `bias` và các lớp chuẩn hóa)
- Hạt giống ngẫu nhiên: 42 (`seed=42`)
- Số epoch tối đa: 30
- Dừng sớm: 7 (`patience=7`)

Lệnh thực thi:

```bash
python -m models.<model>.train --dataset <dataset>
```

#### Cấu hình mặc định theo mô hình:

| Mô hình | `lr` | `batch_size` | `accum_steps` |
| :--- | :---: | :---: | :---: |
| `roberta` | `3e-5` | `8` | `1` |
| `bros` | `3e-5` | `4` | `2` |
| `pick` | `1e-4` | `4` | `2` |
| `layoutlmv3` | `3e-5` | `4` | `2` |
| `donut` | `3e-5` | `2` | `4` |
| `qwen_vl` | `1e-4` | `1` | `8` |

#### Bảng tham số dòng lệnh:

| Tham số | Mặc định | Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--dataset` | `"all"` | `sroie`, `cord_v2`, `wildreceipt`, `all` | Tập dữ liệu huấn luyện |
| `--epochs` | `30` | Số nguyên dương | Số epoch tối đa |
| `--batch_size` | Theo mô hình | Số nguyên dương | Kích thước batch cục bộ |
| `--accum_steps` | Theo mô hình | Số nguyên dương | Số bước tích lũy gradient |
| `--lr` | Theo mô hình | Số thực dương | Tốc độ học khởi tạo |
| `--patience` | `7` | Số nguyên dương | Số epoch dừng sớm |

Kết quả lưu tại `results/<dataset>/<model>/`.

### Bước 4: So sánh kết quả
Tổng hợp kết quả các mô hình tại `results/<dataset>/`:

```bash
python -m tools.compare
```

Lệnh tự động trích xuất bảng `comparison.md` và biểu đồ `comparison.png` theo field-level F1 cho từng tập dữ liệu.

### Thực nghiệm trên Jupyter Notebook
Mỗi mô hình có notebook tương ứng tại `notebooks/<model>_model.ipynb`, hỗ trợ thực thi trên máy cục bộ và Kaggle.