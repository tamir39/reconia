# Reconia

So sánh 6 mô hình trích xuất thông tin hóa đơn trên SROIE, CORD-v2 và WildReceipt. Nhánh [dev](https://github.com/tamir39/reconia/tree/dev) chứa mã huấn luyện; `main` bổ sung demo local.

## Chạy demo

Cần Python 3.12 hoặc `uv` trên Windows.

1. Chạy `setup.bat` để cài môi trường.
2. Chạy `run.bat`, mở <http://127.0.0.1:8766>.
3. Chọn mô hình và dataset, xem mẫu test hoặc tải ảnh lên rồi bấm **Chạy**.

Demo hỗ trợ **RoBERTa, BROS, PICK, LayoutLMv3, Donut và Qwen2.5-VL**. Năm mô hình đầu có ví dụ dự đoán đã lưu; ảnh test Qwen hiển thị **Chưa chạy** cho đến khi thực hiện suy luận. Heatmap cho một trường dự đoán có trên PICK, LayoutLMv3 và Donut.

## Checkpoint

Có **18 checkpoint của 6 mô hình**, tải từ output Kaggle bằng tài khoản có quyền truy cập:

```powershell
.venv/Scripts/python.exe -m pip install -r demo/requirements-assets.txt
.venv/Scripts/python.exe -m demo.fetch_assets --model layoutlmv3
.venv/Scripts/python.exe -m demo.fetch_assets --model qwen_vl
```

Bỏ `--model` để tải tất cả; thêm `--dataset sroie` để chọn một bộ hoặc `--list` để xem danh sách. File được đặt tại `checkpoints/<dataset>/<model>/best_weights.pt`. Có thể đổi thư mục bằng `RECONIA_CHECKPOINT_DIR`. Lệnh chỉ tải trọng số đã huấn luyện.

**Qwen:** dùng adapter của Thịnh, lấy từ SROIE **v1**, CORD-v2 **v3**, WildReceipt **v5**. Chạy lại `setup.bat` để cài thư viện Qwen, rồi tải adapter và mô hình nền (khoảng 7 GB):

```powershell
.venv/Scripts/python.exe -m demo.fetch_assets --model qwen_vl --base
```

Qwen đọc ảnh trực tiếp cho cả mẫu test và ảnh tải lên. Dataset đang chọn quyết định adapter và các trường trích xuất. Mô hình được nạp 4-bit và dùng lại khi đổi dataset; chạy trên CPU có thể mất nhiều thời gian. Giá trị chuẩn của mẫu test chỉ dùng để đối chiếu, không đưa vào đầu vào mô hình.

Checkpoint và mô hình nền lưu ngoài Git. Nguồn và phiên bản nằm trong `demo/data/checkpoints.json`.

## Huấn luyện

Dùng môi trường riêng theo `requirements.txt` và các notebook trong `notebooks/`. Dữ liệu gốc: [vhngthnh/reconia-data](https://www.kaggle.com/datasets/vhngthnh/reconia-data).

```bash
pip install -r requirements.txt
python -m prepare.prepare_sroie
python -m tools.download_weights --model roberta
python -m models.roberta.train --dataset sroie
```

JSONL ở `.cache/<dataset>/`, pretrained weights ở `weights/<model>/`, kết quả ở `results/<dataset>/<model>/`. Mặc định tối đa 25 epoch, patience 5, seed 42; chọn checkpoint theo validation F1 rồi chấm test. Qwen dùng QLoRA và cần môi trường CUDA phù hợp.

## Kết quả

Test Micro-F1 (%), khớp trường–giá trị sau chuẩn hóa chữ thường/khoảng trắng và giữ giá trị lặp.

| Mô hình | SROIE | CORD-v2 | WildReceipt |
|---|---:|---:|---:|
| RoBERTa | 86.75 | 94.45 | 89.40 |
| BROS | 87.68 | 95.80 | 94.34 |
| PICK | 57.58 | 75.43 | 79.99 |
| LayoutLMv3 | 88.08 | 95.48 | 93.81 |
| Donut | 77.76 | 72.40 | 56.93 |
| Qwen2.5-VL | 78.46 | 81.92 | 74.46 |

Output: [RoBERTa](https://www.kaggle.com/code/vhngthnh/roberta-model), [BROS](https://www.kaggle.com/code/vhngthnh/bros-model), [PICK](https://www.kaggle.com/code/vhngthnh/pick-model), [LayoutLMv3](https://www.kaggle.com/code/vhngthnh/layoutlmv3-model), [Donut](https://www.kaggle.com/code/vhngthnh/donut-model), [Qwen](https://www.kaggle.com/code/vhngthnh/qwen-vl-model).

Donut/Qwen dùng giới hạn sinh phụ thuộc độ dài nhãn chuẩn khi đánh giá. CORD-v2 có một cặp train–test và hai cặp train–validation trùng OCR/nhãn. Điểm số giữ theo quy trình và split của output gốc. Ảnh ngoài demo dùng OCR tự động và giới hạn sinh cố định, nên không đồng nhất với phép chấm test.

## Thư mục

```text
models/          Mã 6 mô hình
notebooks/       Notebook huấn luyện
prepare/, utils/ Chuẩn bị dữ liệu và xử lý chung
tools/           Tải mô hình nền, tổng hợp kết quả
demo/
  app.py         HTTP server
  inference.py   Nạp checkpoint và dự đoán
  heatmap.py     Giải thích bằng che vùng
  occlusion.py   Tính điểm sau khi che vùng
  fetch_assets.py Tải checkpoint
  static/        Giao diện
  data/          Mẫu test, nhãn và danh sách checkpoint
  model_assets/  Config và tokenizer
  sample_images/ Ảnh minh họa
```
