# Reconia

Nghiên cứu và so sánh thực nghiệm các phương pháp học sâu cho bài toán trích xuất thông tin hóa đơn (Invoice Information Extraction - IIE).

## 1. Phạm vi nghiên cứu

- **3 tập dữ liệu:** `sroie`, `cord_v2`, `wildreceipt`.
- **6 mô hình:**
  1. `roberta`: Xử lý chuỗi văn bản thuần túy.
  2. `bros`: Kết hợp văn bản và tọa độ 2D.
  3. `pick`: Mạng đồ thị kết hợp văn bản, tọa độ 2D và hình ảnh.
  4. `layoutlmv3`: Đa phương thức kết hợp văn bản, tọa độ 2D và hình ảnh.
  5. `donut`: Biến đổi trực tiếp từ ảnh sang văn bản cấu trúc (OCR-free).
  6. `qwen2_5_vl`: Thị giác - ngôn ngữ lớn (VLM) theo câu lệnh chỉ dẫn.
- **Nguyên tắc so sánh:** Đánh giá độc lập trên từng tập dữ liệu, không so sánh chéo.

## 2. Cấu trúc thư mục

```
reconia/
├── .cache/
│   └── <dataset>/
├── app/
├── docs/
│   └── policy.md
├── evaluate/
│   └── metrics.py
├── models/
│   └── <model>/
│       ├── dataset.py
│       ├── model.py
│       └── train.py
├── notebooks/
│   └── <model>.ipynb
├── prepare/
│   ├── prepare_cord_v2.py
│   ├── prepare_sroie.py
│   └── prepare_wildreceipt.py
├── results/
│   └── <dataset>/
│       ├── <model>/
│       │   ├── best.pt
│       │   ├── history.csv
│       │   ├── last.pt
│       │   ├── loss.png
│       │   ├── metrics.png
│       │   ├── test_metrics.json
│       │   └── train.log
│       ├── comparison.md
│       └── comparison.png
├── tools/
│   ├── common.py
│   ├── compare.py
│   ├── download_weights.py
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

Tải 3 bộ dữ liệu gốc và đặt tại thư mục ngoài song song với `reconia` (hoặc `/kaggle/input/datasets/vhngthnh/reconia-data/` trên Kaggle):
- **SROIE:** [ICDAR 2019 Robust Reading Challenge](https://rrc.cvc.uab.es/?ch=13).
- **CORD-v2:** [clovaai/cord](https://github.com/clovaai/cord).
- **WildReceipt:** [MMOCR KIE Dataset](https://github.com/open-mmlab/mmocr/blob/12558969ee6f11b469a8e964a57730d5df3df6b1/docs/en/datasets/kie.md).

Chuẩn bị thư mục dữ liệu cho SROIE và tải CORD-v2:

```bash
python -m tools.setup_data --dataset <dataset>
```

| Tham số | Mặc định | Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--dataset` | `"all"` | `sroie`, `cord_v2`, `all` | Tập dữ liệu cần khởi tạo hoặc toàn bộ |

### Cấu trúc thư mục dữ liệu gốc

```
../ (thư mục cha song song với reconia)
├── sroie/
│   ├── 0325updated.task1train(626p)/
│   ├── 0325updated.task2train(626p)/
│   ├── SROIE_test_gt_task_3/
│   ├── SROIE_test_images_task_3/
│   └── text.task1&2-test（361p)/
├── cord_v2/
│   ├── dataset_dict.json
│   ├── train/
│   ├── validation/
│   └── test/
└── wildreceipt/
    ├── class_list.txt
    ├── dict.txt
    ├── image_files/
    ├── test.txt
    └── train.txt
```

## 5. Quy trình thực nghiệm

### Bước 1: Chuẩn bị dữ liệu
Chuyển đổi dữ liệu gốc sang JSONL trong `.cache/<dataset>/`:

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
| `--model` | `"all"` | `roberta`, `bros`, `layoutlmv3`, `donut`, `qwen2_5_vl`, `all` | Tên mô hình cần tải hoặc toàn bộ |

### Bước 3: Huấn luyện và đánh giá

Thiết lập chung:
- Kích thước batch hiệu dụng: 8 (`batch_size * accum_steps = 8`)
- Bộ tối ưu hóa: AdamW (`weight_decay=0.01`, loại trừ `bias` và các lớp chuẩn hóa)
- Hạt giống ngẫu nhiên: 42 (`seed=42`)
- Số epoch tối đa: 30
- Dừng sớm: 5 (`patience=5`)

Lệnh thực thi:

```bash
python -m models.<model>.train --dataset <dataset>
```

#### Cấu hình theo mô hình:

| Mô hình | `lr` | `batch_size` | `accum_steps` | Batch hiệu dụng | `epochs` | `patience` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `roberta` | `3e-5` | `8` | `1` | 8 | 30 | 5 |
| `bros` | `3e-5` | `4` | `2` | 8 | 30 | 5 |
| `layoutlmv3` | `2e-5` | `4` | `2` | 8 | 30 | 5 |
| `donut` | `3e-5` | `2` | `4` | 8 | 30 | 5 |
| `pick` | `lr_backbone = 1e-4`<br>`lr_scratch = 5e-4` | `4` | `2` | 8 | 30 | 5 |
| `qwen2_5_vl` | `1e-4` | `1` | `8` | 8 | 30 | 5 |

#### Bảng tham số dòng lệnh:

| Tham số | Mặc định | Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--dataset` | `"all"` | `sroie`, `cord_v2`, `wildreceipt`, `all` | Tập dữ liệu huấn luyện |
| `--epochs` | `30` | Số nguyên dương | Số epoch tối đa |
| `--batch_size` | Theo mô hình | Số nguyên dương | Kích thước batch cục bộ |
| `--accum_steps` | Theo mô hình | Số nguyên dương | Số bước tích lũy gradient |
| `--lr` | Theo mô hình | Số thực dương | Tốc độ học khởi tạo |
| `--patience` | `5` | Số nguyên dương | Số epoch dừng sớm |

Kết quả lưu tại `results/<dataset>/<model>/`.

### Bước 4: So sánh kết quả
Tổng hợp kết quả các mô hình tại `results/<dataset>/`:

```bash
python -m tools.compare
```

Lệnh tự động trích xuất bảng `comparison.md` và biểu đồ `comparison.png` theo field-level F1 cho từng tập dữ liệu.

### Thực nghiệm trên Jupyter Notebook
Mỗi mô hình có notebook tương ứng tại `notebooks/<model>.ipynb`, hỗ trợ thực thi trên máy cục bộ và Kaggle.