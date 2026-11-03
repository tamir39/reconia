# Reconia

So sánh sáu model trích xuất thông tin từ hóa đơn trên SROIE, CORD-v2 và WildReceipt.

## Repo gồm những gì?

```text
models/        Code của 6 model: dataset.py, train.py
notebooks/     Mỗi model một notebook Kaggle
prepare/       Chuyển dữ liệu gốc sang JSONL và mốc dữ liệu baseline
utils/         Hàm dữ liệu, chấm điểm và tiện ích chung
tools/         Tải trọng số, tổng hợp kết quả
tests/         Kiểm tra code và tính tương thích với baseline
reports/       Báo cáo, ghi chú và bằng chứng đối chiếu
```

Các thư mục local không đưa lên Git:

```text
data/raw/      Dữ liệu gốc: sroie/, cord_v2/, wildreceipt/
.cache/        Prepared JSONL, runtime kiểm tra và bằng chứng đã lưu
weights/       Trọng số/tokenizer pretrained
checkpoints/   Checkpoint theo <dataset>/<model>/
results/       Kết quả theo <dataset>/<model>/
```

## Trạng thái

- RoBERTa, BROS, PICK: đã hoàn thành 9 lượt trên ba dataset; giữ nguyên checkpoint/kết quả cũ.
- LayoutLMv3, Donut, Qwen: code đã đối chiếu dữ liệu/cách chấm với các lượt cũ; cần thử cấu hình thật trên GPU rồi train.
- Source và tokenizer của lượt cũ được lưu riêng để không nhầm với code local đã chỉnh.

| Model | Đầu vào | Cách học | Batch × tích lũy | LR |
|---|---|---|---:|---:|
| RoBERTa | Text annotation | BIO | 8 × 1 | 3e-5 |
| BROS | Text + box annotation | BIO | 4 × 2 | 3e-5 |
| PICK | Ảnh + text + box | Đồ thị + BiLSTM/CRF | 4 × 2 | 1e-4 |
| LayoutLMv3 | Ảnh + text + box | BIO, tắt OCR | 4 × 2 | 3e-5 |
| Donut | Ảnh | Sinh JSON | 1 × 8 | 3e-5 |
| Qwen2.5-VL-3B | Ảnh + prompt tên trường | LoRA sinh JSON | 1 × 8 | 1e-4 |

## Chạy ba model mới

Cài môi trường: `pip install -r requirements.txt`.

**Kaggle:** mở notebook tương ứng trong `notebooks/`, gắn dataset
[`vhngthnh/reconia-data`](https://www.kaggle.com/datasets/vhngthnh/reconia-data), chọn `DATASET`, chạy lần lượt sáu mục.
Dùng source mới trên nhánh `dev`; notebook kiểm tra đúng nhánh,
in commit đang dùng và không tự cập nhật checkout đã tồn tại.

**Local:** đặt dữ liệu ở `data/raw/<dataset>/`, hoặc đặt `RECONIA_DATA_DIR`
tới thư mục chứa ba dataset. Ví dụ LayoutLMv3/SROIE:

```bash
python -m prepare.prepare_sroie
python -m tools.download_weights --model layoutlmv3
python -m models.layoutlmv3.train --dataset sroie
```

Đổi `layoutlmv3` thành `donut` hoặc `qwen_vl`; dataset là `sroie`, `cord_v2`, `wildreceipt`.
Sau khi chốt thí nghiệm mới chạy test:

```bash
python -m models.layoutlmv3.train --dataset sroie --test-only
python -m tools.compare
```

`--resume` tiếp tục checkpoint tương thích. Run mới không ghi đè checkpoint đã có.
Checkpoint Qwen chỉ lưu LoRA, cần cùng base weights để nạp lại.

## Dữ liệu và cách chấm

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| SROIE | 501 | 125 | 347 |
| CORD-v2 | 800 | 100 | 100 |
| WildReceipt | 1014 | 253 | 472 |

Không chạy OCR mới và không dùng dữ liệu OCR đã sửa/lọc của các thử nghiệm cũ.
SROIE/WildReceipt tách 20% train làm validation với seed 42; CORD giữ split có sẵn.
Ba trainer mới xác minh toàn bộ prepared data với mốc các lượt đã chạy trước khi train.

Tất cả model trả về `{"field": ["value", ...]}`. Chấm khớp chính xác trong cùng trường,
sau chuẩn hóa chữ thường và khoảng trắng; giữ giá trị lặp, lấy Macro Precision/Recall/F1 theo trường.
Best checkpoint chọn bằng F1 validation. Không dùng GT làm input suy luận.
LayoutLMv3 lấy tối đa 512 token đầu như RoBERTa/BROS đã chạy; Donut/Qwen đọc ảnh đầy đủ.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Chi tiết mô hình/bài báo: [ghi chú train](reports/training-notes.md).
Dữ liệu và scorer đã đối chiếu: [kết quả baseline](reports/completed-baseline-alignment-20261027.json).
Checkpoint cũ cần source cùng lượt, không giả định tương thích với mọi thay đổi ở code local.
