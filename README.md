# Reconia

Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin từ hóa đơn.

## Dữ liệu và cách đánh giá

Dùng ảnh, text/box và GT từ dữ liệu gốc SROIE, CORD-v2, WildReceipt.
Không chạy PaddleOCR, không áp dụng các bản sửa GT hoặc lọc mẫu của thí nghiệm OCR trước.
Giữ train/test gốc; SROIE và WildReceipt tách 20% train làm validation với seed 42.
CORD giữ validation có sẵn.

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| SROIE | 501 | 125 | 347 |
| CORD-v2 | 800 | 100 | 100 |
| WildReceipt | 1014 | 253 | 472 |

Các số trên được kiểm tra với bộ raw local. Nếu dùng gói raw khác, kiểm tra lại số mẫu.

Mọi model trả về cùng dạng `{"field": ["value", ...]}`.
Một giá trị được tính đúng khi khớp GT trong cùng trường, sau chuẩn hóa chữ thường và khoảng trắng.
Tính Precision, Recall, F1 từng trường rồi lấy trung bình (Macro).
Validation chạy sau mỗi epoch và chọn checkpoint có F1 cao nhất. Test chạy riêng khi đã chốt thí nghiệm.

## Sáu model

| Model | Input | Cách học | Batch × tích lũy | LR |
|---|---|---|---:|---:|
| RoBERTa | Text gốc | Phân loại token BIO | 8 × 1 | 3e-5 |
| BROS | Text + box gốc | Phân loại token BIO | 4 × 2 | 3e-5 |
| PICK | Ảnh + text + box gốc | Encoder, đồ thị, BiLSTM/CRF | 4 × 2 | 1e-4 |
| LayoutLMv3 | Ảnh + text + box gốc | Phân loại token BIO; tắt OCR trong processor | 4 × 2 | 3e-5 |
| Donut | Ảnh | Fine-tune encoder–decoder sinh JSON | 1 × 8 | 3e-5 |
| Qwen2.5-VL-3B | Ảnh + yêu cầu tên trường | LoRA sinh JSON | 1 × 8 | 1e-4 |

Chung: seed 42, AdamW, tối đa 30 epoch, tối thiểu 5 epoch, patience 5.
Bộ đếm không cải thiện bắt đầu từ epoch 5. AMP ghi lại từng lần bỏ cập nhật.

Donut dùng ảnh 1280 × 960, tối đa 1535 token đầu ra theo giới hạn decoder.
Qwen dùng LoRA r=16, alpha=32, dropout=0.05 trên attention của phần ngôn ngữ;
ảnh từ 256 đến 768 visual token, đầu ra tối đa 2048 token.
Text dài của RoBERTa/BROS/LayoutLMv3 được chia cửa sổ 512 token, chồng 64 token,
rồi ghép dự đoán về tài liệu. Token lặp, token đặc biệt và padding không tính loss.

## File nằm ở đâu?

```text
prepare/                    Chuyển annotation gốc sang JSONL
models/<model>/             Adapter và trainer
notebooks/<model>_model.ipynb
utils/                      Hàm dữ liệu, metric và vòng train dùng chung
.cache/<dataset>/
    train.jsonl
    val.jsonl
    test.jsonl
    labels.json
weights/<model>/            Trọng số pretrained
checkpoints/<dataset>/<model>/
results/<dataset>/<model>/  history.csv, train.log, biểu đồ, test_metrics.json
```

Trong JSONL: `image_path` là ảnh, `ocr` là **text/box của annotation gốc** kèm nhãn BIO,
`gt` là đáp án để chấm điểm. Tên khóa `ocr` giữ theo code của Thịnh; không có nghĩa là chạy PaddleOCR.
GT chỉ làm target khi học và đáp án khi đánh giá, không đưa vào input suy luận.

## Chạy

Cài `pip install -r requirements.txt`.

Dữ liệu raw: [vhngthnh/reconia-data](https://www.kaggle.com/datasets/vhngthnh/reconia-data).
Trên máy local, đặt `RECONIA_DATA_DIR` là thư mục chứa ba bộ dữ liệu.
Trên Kaggle, thêm dataset này vào Input. Có thể đặt `RECONIA_PREPARED_DIR` nếu muốn đổi nơi lưu JSONL.

Ví dụ với LayoutLMv3/SROIE:

```bash
python -m prepare.prepare_sroie
python -m tools.download_weights --model layoutlmv3
python -m models.layoutlmv3.train --dataset sroie
```

Chấm test sau khi đã chốt thí nghiệm:

```bash
python -m models.layoutlmv3.train --dataset sroie --test-only
python -m tools.compare
```

Thay tên model/dataset để chạy các trường hợp khác. PICK không cần bước tải weights ở trên;
ResNet tải trọng số qua torchvision khi khởi tạo.

Mỗi notebook chọn một `DATASET`; cell setup giữ nguyên source và kết quả khi chạy lại.
Để lấy code mới, cập nhật checkout chủ động. Notebook dùng nhánh `codex/original-data-six-models`.
Notebook có thể cần Kaggle Secret `GH_TOKEN` để clone repo private; token không được ghi vào Git remote.
Cell test mặc định `RUN_TEST=False`.

`--resume` nối tiếp run đã có. Chạy mới sẽ báo lỗi nếu thư mục đó đã có checkpoint;
chuyển run cũ sang nơi khác trước khi chạy mới.
Không dùng checkpoint PaddleOCR làm kết quả của thí nghiệm text/box gốc.

## Kiểm tra code

```bash
python -m unittest discover -s tests -p test_pipeline.py -v
```

Kiểm tra local chưa thay thế chạy thử GPU với trọng số đầy đủ. Xem
[biên bản kiểm tra](reports/original-data-integration-20261023.md).
