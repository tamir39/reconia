# Reconia

Nghiên cứu và so sánh thực nghiệm các phương pháp Học sâu cho bài toán Trích xuất thông tin hóa đơn (Invoice Information Extraction - IIE).

## 1. Phạm vi nghiên cứu

- **3 bộ dữ liệu:** `sroie`, `cord_v2`, `wildreceipt`.
- **6 mô hình:**
  1. `roberta`: Mô hình tiền huấn luyện ngôn ngữ chỉ tiếp nhận chuỗi văn bản thuần túy, không sử dụng đặc trưng bố cục không gian hay thị giác.
  2. `bros`: Mô hình mã hóa kết hợp văn bản và tọa độ 2D, mô hình hóa mối quan hệ không gian tương đối giữa các hộp từ.
  3. `pick`: Mô hình học biểu diễn đồ thị không gian, xây dựng đồ thị liên kết giữa các hộp từ để gán nhãn thực thể.
  4. `layoutlmv3`: Mô hình đa phương thức kết hợp đồng thời ba luồng thông tin gồm văn bản, bố cục tọa độ 2D và hình ảnh tài liệu.
  5. `donut`: Mô hình biến đổi tuần tự từ ảnh sang văn bản cấu trúc, không phụ thuộc vào hệ thống OCR trung gian.
  6. `qwen2_5_vl`: Mô hình thị giác và ngôn ngữ lớn, trích xuất cấu trúc thực thể trực tiếp từ ảnh tài liệu theo câu lệnh chỉ dẫn.
- **Nguyên tắc so sánh:** Mỗi bộ dữ liệu đại diện cho một không gian bài toán độc lập với đặc thù cấu trúc và nhãn riêng biệt; việc huấn luyện và so sánh giữa 6 mô hình được thực hiện độc lập trên từng bộ dữ liệu, không so sánh chéo kết quả giữa các bộ dữ liệu khác nhau.

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

Tải 3 bộ dữ liệu gốc từ các liên kết chính thức và đặt tại thư mục ngoài song song với `reconia` (hoặc trên Kaggle tại `/kaggle/input/datasets/vhngthnh/reconia-data/`):

- **SROIE:** Tải từ cuộc thi ICDAR 2019 Task 1, Task 2 và Task 3 tại [ICDAR 2019 Robust Reading Challenge](https://rrc.cvc.uab.es/?ch=13).
- **CORD-v2:** Tải từ kho lưu trữ chính thức của Clova AI tại [clovaai/cord](https://github.com/clovaai/cord).
- **WildReceipt:** Tải theo hướng dẫn của MMOCR tại [MMOCR KIE Dataset Documentation](https://github.com/open-mmlab/mmocr/blob/12558969ee6f11b469a8e964a57730d5df3df6b1/docs/en/datasets/kie.md).

Đối với SROIE và CORD-v2, thực thi script sau để tự động tải CORD-v2 từ Hugging Face và sắp xếp thư mục SROIE trước khi chuẩn bị dữ liệu:

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
Chuyển đổi dữ liệu gốc sang định dạng JSONL trong `.cache/<dataset>/`:

```bash
python -m prepare.prepare_sroie
python -m prepare.prepare_cord_v2
python -m prepare.prepare_wildreceipt
```

### Bước 2: Tải trọng số mô hình
Tải trước trọng số pretrained từ Hugging Face về thư mục `weights/<model>/`:

```bash
python -m tools.download_weights --model <model>
```

| Tham số | Mặc định | Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--model` | `"all"` | `roberta`, `bros`, `layoutlmv3`, `donut`, `qwen2_5_vl`, `all` | Tên mô hình cần tải hoặc toàn bộ |

### Bước 3: Huấn luyện và đánh giá
Kích thước batch hiệu dụng được đồng bộ ở mức 16 giữa các mô hình (như batch size 8 kết hợp tích lũy gradient 2 bước). Quá trình tối ưu hóa tuân thủ cấu hình baseline của bài báo gốc, loại trừ suy giảm trọng số đối với hệ số chệch và các lớp chuẩn hóa. Các tham số `--lr` và `--batch_size` được điều chỉnh theo từng kiến trúc.

```bash
python -m models.<model>.train --dataset <dataset>
```

Cấu hình mẫu cho RoBERTa:

| Tham số | Mặc định | Lựa chọn | Mô tả |
| :--- | :---: | :--- | :--- |
| `--dataset` | `"all"` | `sroie`, `cord_v2`, `wildreceipt`, `all` | Tập dữ liệu huấn luyện |
| `--epochs` | `20` | Số nguyên dương | Số epoch tối đa |
| `--batch_size` | `8` | Số nguyên dương | Kích thước batch |
| `--accum_steps` | `2` | Số nguyên dương | Số bước tích lũy gradient |
| `--lr` | `3e-5` | Số thực dương | Tốc độ học khởi tạo |
| `--patience` | `5` | Số nguyên dương | Số epoch chờ dừng sớm |

Kết quả và checkpoint được lưu tại `results/<dataset>/<model>/`.

### Bước 4: So sánh kết quả
Tổng hợp kết quả giữa các mô hình đã chạy và lưu vào `results/<dataset>/`:

```bash
python -m tools.compare
```

Lệnh tự động trích xuất bảng số liệu `comparison.md` và vẽ biểu đồ so sánh `comparison.png` cho từng tập dữ liệu. Chỉ số đánh giá được tính toán theo cấp trường (field-level F1): so khớp chính xác sau khi chuẩn hóa đối với trường đơn trị và thuật toán ghép cặp 1-1 đối với trường đa trị.

### Thực nghiệm qua Jupyter Notebook
Mỗi mô hình có sổ tay thực nghiệm tương ứng tại `notebooks/<model>.ipynb`, hỗ trợ thực thi trên máy cục bộ và Kaggle.