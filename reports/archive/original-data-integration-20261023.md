# Tích hợp dữ liệu gốc và sáu model — 23/10/2026

Nhánh: `codex/original-data-six-models`, lấy nền từ commit `4b8d36a` của Thịnh.
Nhánh `codex/cleanup-reconia-20261021` giữ nguyên để quay lại.

## Đã làm

- Giữ hướng đơn giản của Thịnh: raw → JSONL → adapter → train/val; test chạy riêng.
- Hoàn thiện adapter/trainer/notebook LayoutLMv3, Donut và Qwen2.5-VL.
- Sửa tokenizer RoBERTa. Dùng chung xử lý cửa sổ text dài cho RoBERTa/BROS/LayoutLMv3.
- Cell setup không xóa repo/checkpoint; Git token chỉ truyền qua môi trường khi clone.
- Sửa loss của nhóm gradient accumulation cuối; giữ log từng AMP skip.
- PICK xử lý box padding an toàn và tính loss nhạy với FP16 bằng float32.
- Nhãn BIO đặt cùng JSONL; đóng đủ cặp B/I của mỗi field.
- Không dùng PaddleOCR hoặc các patch GT đã sửa. Các dữ liệu, cache OCR, checkpoint và backup cũ không được ghi đè.
- Cùng Precision/Recall/F1 theo giá trị field; lấy trung bình từng field (Macro), chọn best bằng F1 validation.

## Dữ liệu được kiểm tra

Nguồn local: `.cache/preprocessing-complete/raw`.
Bản chuyển đổi mới: `.cache/sroie`, `.cache/cord_v2`, `.cache/wildreceipt`.

| Bộ | Train | Val | Test |
|---|---:|---:|---:|
| SROIE | 501 | 125 | 347 |
| CORD-v2 | 800 | 100 | 100 |
| WildReceipt | 1014 | 253 | 472 |

Giữ tập gốc; SROIE/WildReceipt tách validation 20% với seed 42, CORD dùng validation chính thức.
Các số này khác thí nghiệm OCR đã lọc mẫu. Không gộp kết quả hai điều kiện thành cùng một bảng benchmark.
Đã đối chiếu 5.705 file raw với gói Kaggle của Thịnh. Đối chiếu nguồn và 347 GT test
SROIE đã được ghi riêng trong [báo cáo nguồn dữ liệu](official-dataset-comparison-20261023.md).

## Kiểm tra đã qua

- Sáu notebook: cú pháp cell Python và import trainer đầy đủ.
- 18 tổ hợp model × dataset: adapter/collator trên mẫu validation đầu và cuối;
  tensor hữu hạn, padding đúng kiểu. Không phải chạy toàn bộ tài liệu qua model.
- Donut/Qwen: thay GT trong chế độ inference không làm thay đổi input; batch inference không chứa labels.
- Kiểm tra tất cả đường dẫn ảnh, số mẫu, giới hạn PICK trên train/val/test.
- Target train/val dài nhất: Donut 769 token, Qwen 701 token; nằm trong giới hạn.
- Forward/loss/reload với model khởi tạo ngẫu nhiên thu nhỏ của RoBERTa, BROS,
  LayoutLMv3, Donut, Qwen; PICK dùng kiến trúc hiện tại và ResNet ngẫu nhiên.
- Donut/Qwen sinh được token; LoRA Qwen lưu/nạp lại được.
- 5 unit test qua: metric và giá trị lặp, JSON lỗi, cửa sổ dài, accumulation cuối,
  checkpoint lưu/nạp và chống ghi đè.

Runtime local: Python 3.12, PyTorch 2.6 CPU, Transformers 4.57.6, PEFT 0.17.1.
Kết quả kiểm tra chi tiết lưu local tại `.cache/original-data-preflight.json`
và `.cache/original-model-checks.json`.

## Rà lại sau khi chốt dữ liệu gốc

- Cả sáu notebook dùng `vhngthnh/reconia-data`, chọn một trong ba bộ qua `DATASET`.
  Không có đường dẫn sang gói PaddleOCR. Khi chạy local cần đặt `RECONIA_DATA_DIR`
  tới `.cache/preprocessing-complete/raw`; môi trường hiện tại chưa đặt biến này.
- Kiểm tra cú pháp, đường truyền tên dataset tới prepare/trainer và cell xem mẫu
  cho đủ 18 tổ hợp. Lệnh prepare/train được mock trong phép kiểm tra notebook;
  cell xem mẫu đọc ảnh và JSONL thật. Không thực thi tải weights hoặc train.
- Đã đọc lại 3.712 mẫu: mọi đường dẫn ảnh tồn tại; GT chuyển đổi khớp GT raw theo
  quy tắc hiện tại (SROIE giữ bốn trường, CORD flatten `gt_parse`, WildReceipt lấy
  các nhãn `_value`). Tất cả trường GT có trong bộ nhãn. Chia validation và tập
  test khớp cấu hình ở bảng trên. Không sửa hoặc làm sạch dữ liệu nguồn.
- Cả sáu trainer dùng cùng `utils.metrics.evaluate_dataset`: Precision/Recall/F1
  từ giá trị khớp trong cùng trường sau chuẩn hóa chữ thường/khoảng trắng,
  rồi trung bình theo trường (Macro). Giá trị lặp được ghép một-một.
  Đây là cách đánh giá thống nhất cho đề tài, không khẳng định trùng protocol
  leaderboard của cả ba bộ dữ liệu.
- Chọn best bằng Macro-F1 validation sau mỗi epoch; cell test vẫn `RUN_TEST=False`.
- Chỉ bổ sung đường dẫn raw và cell xem ảnh/text/box/GT mẫu trong notebook,
  làm rõ mô tả metric và sửa số thứ tự mục PICK. Không đổi thuật toán metric/trainer.
- Chạy lại 5 unit test: đạt. Lần đầu hai test lỗi quyền ghi thư mục tạm của sandbox;
  đặt TEMP/TMP vào `.cache/notebook-review-temp` rồi cả năm đạt, không cần sửa code.
- 12 file JSONL/labels hiện có giữ nguyên nội dung. Kiểm tra và mẫu xem được lưu tại
  `.cache/notebook-review-20261023.json`; README chỉ rõ vị trí mẫu trong dataset Kaggle.

## Trước khi train

Chưa chạy fine-tune trên dữ liệu hóa đơn, chưa chấm điểm test, chưa chạy Kaggle.
Các phép thử model nhỏ xác nhận đường đi của code; chưa xác nhận VRAM, tốc độ hoặc
ổn định AMP của sáu model đầy đủ trên GPU. Bước tiếp theo là smoke GPU trước train chính.
