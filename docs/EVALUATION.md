# Quy trình đánh giá — cập nhật 25/09/2026

## Phạm vi và ba loại chỉ số

Phạm vi đã thống nhất: GNN, RoBERTa, LayoutLMv3, LiLT, Qwen2.5-VL và InternVL. Hai họ VLM do Thịnh đề xuất; kích thước/checkpoint cụ thể cần thử bộ nhớ. Hiện chỉ GNN và RoBERTa có đường chạy huấn luyện/suy luận.

1. **Giá trị trường dùng chung:** `field-value-nfkc-casefold-v1` trong `app/evaluation/values.py`. Ghép chính xác `(fieldtype, text)` trong từng tài liệu sau chuẩn hóa Unicode NFKC, chữ thường và khoảng trắng. Giữ dấu câu, số lần xuất hiện và mọi trường ground truth, kể cả trường OCR không đọc được. Báo cáo micro P/R/F1, macro F1 và từng loại trường. Chỉ số này không chấm vị trí hoặc nhóm dòng hàng; VLM cần chuyển đầu ra sang cùng schema trước khi được so sánh.
2. **Thực thể OCR:** `ocr-bio-exact-span-v2` chấm loại trường, khoảng từ và trang; không yêu cầu chuỗi văn bản giống nhau. Giữ tài liệu không có nhãn và loại trường chưa xuất hiện khi train. Các trường không gán được vào OCR không có trong ground truth BIO, nên chỉ số này phải tách khỏi chỉ số giá trị trường.
3. **DocILE chính thức:** chạy công cụ của [DocILE Benchmark](https://github.com/rossumai/docile) để chấm KILE/LIR. Chỉ số BIO nội bộ không thay thế phép ghép vị trí và OCR của DocILE. LIR còn cần dự đoán `line_item_id`; pipeline hiện chưa tạo nhóm dòng hàng. Không gắn nhãn “official” cho kết quả nội bộ.

## Dữ liệu và chọn checkpoint

### SROIE: bốn key gốc dùng chung

`app/evaluation/sroie_values.py` cung cấp `sroie-four-field-nfkc-casefold-v1`.
Ground truth lấy từ `field-value-targets.json` đã đối chiếu các file key gốc, không
lấy từ những đoạn OCR được heuristic gán nhãn. Dự đoán của GNN/RoBERTa giữ thứ tự
OCR; ghép tất cả đoạn không rỗng cùng `company`, `date`, `address`, `total` bằng một
dấu cách. Không chọn đoạn theo confidence/ground truth và không bỏ đoạn lặp. VLM
cung cấp một giá trị văn bản mỗi key, qua cùng hàm ghép và chuẩn hóa NFKC/casefold/
khoảng trắng. Giữ dấu câu và nội dung số, không sửa OCR hay chuẩn hóa tiền/date riêng.

Key thiếu chú giải gốc được coi là chưa biết và loại khỏi phép chấm cho mọi mô hình;
dự đoán vẫn được lưu. SROIE test hiện có 251 giá trị có nhãn trên 63 receipt (một
address bị thiếu). Trường có nhãn nhưng OCR không đọc được vẫn tính là FN nếu không
dự đoán đúng. Không loại tài liệu thiếu dự đoán: ID phải khớp toàn bộ split. Chỉ
`--limit` tường minh mới cho phép smoke và khi đó `complete=false`.

```powershell
python -m app.evaluation.sroie_values --predictions reports/predictions/predictions_gnn_sroie_test.json --targets .cache/receipt-prep/sroie/field-value-targets.json --split test --output reports/predictions/gnn_sroie_test.four-field.metrics.json
```

Lưu điểm bốn key ở sidecar riêng, kèm hash predictions/nhãn gốc/dự đoán sau ghép.
Điểm `value_metrics` cũ của SROIE vẫn là đoạn OCR; không dùng nó thay thế điểm bốn
key khi so với VLM. Notebook RoBERTa receipt tự chạy thêm phép chấm này sau suy luận.
Adapter suy luận VLM chưa hiện thực, nhưng định dạng và phép chấm chung đã có.

### Split và chọn checkpoint

| Bộ dữ liệu | Train | Val | Test nội bộ |
|---|---:|---:|---:|
| DocILE | 4.680 | 500 | 500 |
| SROIE | 500 | 63 | 63 |
| CORD-v2 | 800 | 100 | 100 |

DocILE test nội bộ được lấy từ train gốc; không phải test chính thức 1.000 tài liệu ẩn nhãn. Checkpoint từng học trên toàn bộ train gốc không đủ điều kiện chấm test nội bộ này. Chỉ so sánh với số liệu công bố khi trùng tập dữ liệu, tác vụ, cách chia tập và công cụ đánh giá.

Chỉ dùng train để học và xây từ điển nhãn. Dùng val để chọn epoch/tham số, sau đó chạy test với cấu hình đã chốt. Không chọn mô hình theo test. Ghi commit, seed, tham số, phiên bản thư viện, thiết bị và nguồn checkpoint cùng kết quả. Hash split hiện được tính trên danh sách ID theo thứ tự adapter trả về.

## Chạy và lưu bằng chứng

Cấu hình `DOCILE_DIR`, `SROIE_DIR`, `CORD_DIR` trong môi trường hoặc `.env`; không commit dữ liệu, trọng số hay `.env`.

```powershell
python -m app.data.export --dataset docile --output .cache/export-v2/docile
python -m app.training.train_gnn --dataset docile --epochs 20 --device cuda
python -m app.evaluation.predict --model gnn --dataset docile --split test --checkpoint checkpoints/gnn_docile_best.pt
```

Lệnh xuất dữ liệu mặc định có train/val/test, lưu manifest và kiểm tra SHA-256 của từng shard. CORD `validation` được xuất dưới tên `val`. Export DocILE cũ 5.180 train / 500 val phải tạo lại: train cũ chứa các tài liệu nay thuộc test nội bộ.

Mỗi file dự đoán mới có file kèm `*.metrics.json`: số tài liệu thực tế/kỳ vọng, `complete`, protocol và hash checkpoint/split/predictions. Chỉ dùng `complete=true` trên đúng tập dự kiến cho bảng kết quả cuối. `--limit` tạo tên `_sampleN` và không phải một benchmark hoàn chỉnh. Khi tự đặt `--output`, hãy dùng đường dẫn riêng cho smoke test.

GNN trước bản sửa 24/09 dùng Python `hash()`, thay đổi giữa các tiến trình. Checkpoint đó thiếu `text_hash_version` và bị từ chối: **cần train lại với `blake2b-64-v1`**, không sửa metadata để ép nạp trọng số cũ. RoBERTa mới lưu cấu hình backbone và từ điển nhãn trong checkpoint; tokenizer vẫn cần có sẵn hoặc tải từ nguồn mô hình.

Với môi trường đã cài công cụ upstream, kiểm tra KILE trên val công khai:

```powershell
docile_evaluate --task KILE --dataset-path <DOCILE_DATASET_PATH> --split val --predictions <PREDICTIONS_JSON>
```

Kiểm tra schema và tách trường KILE theo yêu cầu upstream trước khi nộp; chưa có kết quả chạy công cụ chính thức trong lần kiểm tra này. Không dùng lệnh trên để diễn giải test nội bộ thành test chính thức.

## Trạng thái bằng chứng hiện tại

Cập nhật 25/09: GNN đã có kết quả đầy đủ và tái chấm field-value trên cả ba bộ; RoBERTa đã kiểm chứng DocILE. Entity F1 GNN: DocILE 66,82%, SROIE 80,77%, CORD 69,73%. RoBERTa/DocILE 79,69%. Bốn key gốc SROIE của GNN đạt F1 54,70%, tách khỏi điểm đoạn OCR. [Bảng đã kiểm chứng](../reports/verified_comparison.md) là nguồn cho tiến độ hiện tại; phần dưới mô tả phát hiện ở các bản lịch sử trước khi chạy lại.

Các báo cáo do Thịnh đẩy ngày 21–22/09 có giá trị tham khảo, chưa được tái lập độc lập. Ba file dự đoán GNN và file RoBERTa/SROIE chỉ có 2 tài liệu; không chứng minh đánh giá toàn bộ test. RoBERTa/CORD có 100 và RoBERTa/DocILE có 500 tài liệu, nhưng vẫn cần checkpoint và tái chấm bằng quy trình đã sửa. Không sửa hoặc suy đoán lại các điểm số cũ.

Kiểm tra GPU cục bộ với 16 train / 4 val chỉ xác nhận đường chạy, không phải kết quả thực nghiệm để đưa vào kết luận. Các bước còn thiếu được ghi trong [MIDTERM.md](MIDTERM.md).
