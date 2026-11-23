# Reconia

Trích xuất thông tin hóa đơn với **5 model**: RoBERTa, BROS, PICK, LayoutLMv3 và Donut trên SROIE, CORD-v2, WildReceipt. Nhánh chuẩn bị nộp: `final-submit`.

Source, nhãn, cấu hình và kết quả được đồng bộ từ output Kaggle của Thịnh đã kiểm chứng ngày **21/11/2026**. Đây là snapshot cố định, không tự nhận output mới khi hash khác. Qwen và baseline sáu model cũ nằm trong `archive/`, không thuộc bảng benchmark này.

## Mở demo

Windows, Python 3.12 hoặc uv:

1. Chạy `setup.bat` lần đầu để cài môi trường `.venv`.
2. Chạy `run.bat`, giữ cửa sổ lệnh mở. App tại <http://127.0.0.1:8766>.

Source có sẵn bảng điểm, 30 ví dụ và ảnh. Để bấm **Chạy** trên mẫu test hoặc ảnh ngoài, cần checkpoint đúng model và dataset. Trọng số lớn không đưa vào Git/ZIP source. Tải bằng tài khoản Kaggle có quyền đọc output, dùng key đã cấu hình:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-assets.txt
.venv/Scripts/python.exe -m tools.fetch_assets --checkpoints
.venv/Scripts/python.exe -m tools.verify_final --weights
```

Có thể tải riêng bằng `--model layoutlmv3`. Tool chỉ đọc output, xác minh SHA-256, không gửi job train. Nếu output nguồn đổi, tool dừng. Hoặc đặt `RECONIA_CHECKPOINT_DIR` tới thư mục checkpoint đã có, cấu trúc `<dataset>/<model>/best_weights.pt`.

Demo chạy inference thật, không train. Mẫu test dùng OCR chú thích; ảnh ngoài dùng RapidOCR với bốn model có OCR, Donut đọc ảnh trực tiếp. Heatmap là **Occlusion cho một trường**, hỗ trợ PICK, LayoutLMv3, Donut. Vùng đỏ hỗ trợ đáp án đang xem, không khẳng định đáp án đúng. Xem [demo/README.md](demo/README.md).

## Kết quả đã chốt

**Micro-F1 (%)**: cộng TP/FP/FN trên các trường và tài liệu; khớp giá trị sau chuẩn hóa chữ thường/khoảng trắng, giữ giá trị lặp.

| Model | SROIE | CORD-v2 | WildReceipt |
|---|---:|---:|---:|
| RoBERTa | 86.75 | 94.45 | 89.40 |
| BROS | 87.68 | 95.80 | 94.34 |
| PICK | 57.58 | 75.43 | 79.99 |
| LayoutLMv3 | 88.08 | 95.48 | 93.81 |
| Donut | 77.76 | 72.40 | 56.93 |

Giá trị đầy đủ, history/log: [reports/benchmarks/](reports/benchmarks/). Nguồn file và thay đổi chạy local: [provenance.json](reports/benchmarks/provenance.json). Không lấy điểm từ báo cáo Macro-F1 cũ. Ví dụ đúng/sai chỉ minh họa một trường, không thay thế đánh giá toàn bộ test.

## Cấu trúc hiện tại

```text
models/              Source 5 model từ các lượt Kaggle đã chốt
notebooks/           5 notebook gọn; mọi cờ chạy mặc định tắt
prepare/             Chuẩn bị raw data, manifest kiểm tra 9 split
utils/               Hàm chung, Micro-F1, biểu đồ
demo/               App, 30 ví dụ, 16 ảnh, nhãn/config/tokenizer
reports/benchmarks/  15 bộ metrics/history/log và nguồn đối chiếu
reports/archive/     Báo cáo, ghi chú, bản thảo lịch sử
tests/              Kiểm tra scorer, dữ liệu, notebook, heatmap
tools/              Tải tài nguyên, kiểm chứng, so sánh, đóng gói
archive/             Source sáu model cũ để tham khảo
run.bat              Mở demo
setup.bat            Cài môi trường demo CPU
```

Không đưa lên Git: `.env`, `.venv/`, `.cache/`, `data/`, `weights/`, `checkpoints/`, `results/`, `output/`, `dist/`. `.cache` chỉ chứa dữ liệu chuẩn bị/cache có thể tạo lại; code và tài nguyên demo nhỏ nằm trong repo.

## Quy trình train tham khảo

**Không cần train để demo/nộp source.** Notebook không train khi mở. Khi chủ động làm thí nghiệm mới, dùng checkout riêng giữ checkpoint đã chốt, GPU đủ bộ nhớ và bản PyTorch phù hợp GPU, rồi bật cờ cần thiết trong notebook.

| Model | Đầu vào | Batch × tích lũy | LR |
|---|---|---:|---:|
| RoBERTa | Chữ OCR | 8 × 1 | 3e-5 |
| BROS | Chữ + box OCR | 8 × 1 | 3e-5 |
| PICK | Ảnh + chữ + box OCR | 4 × 2 | 5e-4 |
| LayoutLMv3 | Ảnh + chữ + box OCR | 4 × 2 | 3e-5 |
| Donut | Ảnh + task token theo dataset | 4 × 2 | 3e-5 |

Default source đã xuất: tối đa 25 epoch, patience 5, seed 42; chọn best validation F1 rồi chấm test. Đây không phải cấu hình Qwen/Donut retrain cũ. Cleanup giữ cách tính loss của lượt xuất và không tạo benchmark mới.

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| SROIE | 501 | 125 | 347 |
| CORD-v2 | 800 | 100 | 100 |
| WildReceipt | 1014 | 253 | 472 |

Raw data tại `data/raw/<dataset>/`, hoặc đặt `RECONIA_DATA_DIR` tới thư mục chứa ba dataset. Có thể tải split đã chốt bằng `python -m tools.fetch_assets --prepared`, hoặc chạy `python -m prepare.prepare_<dataset>`. Trước train, chạy `python -m tools.verify_final --prepared` để kiểm tra số hàng, thứ tự, OCR, box, nhãn của 9 split. Notebook cũng có bước kiểm tra này khi bật train.

Giữ split đã báo cáo. CORD-v2 có ghi nhận một số mẫu trùng nội dung giữa các split trong kiểm toán trước, nên không mô tả test hoàn toàn độc lập. Không lọc trùng rồi gắn lại điểm cũ.

Trainer dừng nếu checkpoint đã tồn tại; `--resume` chỉ để tiếp tục lượt tương thích. `DATASET="all"` trong notebook lặp ba dataset và vẫn chịu giới hạn thời gian Kaggle.

## Kiểm tra và xuất source

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m tools.verify_final
.venv/Scripts/python.exe -m tools.check_demo_predictions
.venv/Scripts/python.exe -m tools.compare
.venv/Scripts/python.exe -m tools.export_submission
```

Kiểm tra prediction nạp 15 checkpoint để chạy lại 30 ví dụ. ZIP source tại `dist/reconia-final-source.zip`, không chứa dữ liệu riêng, môi trường hay trọng số. Bản thảo học thuật trong `reports/archive/` chưa được coi là báo cáo final đã duyệt.
