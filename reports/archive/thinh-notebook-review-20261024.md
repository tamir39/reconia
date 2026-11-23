# Kiểm tra notebook của Thịnh — 24/10/2026

Đã đọc notebook và output thực tế trên Kaggle; không chạy lại train/test. 36/36 ô số trên ảnh slide khớp test_metrics.json khi làm tròn 4 chữ số. Cả ba notebook COMPLETE.

Notebook: [RoBERTa](https://www.kaggle.com/code/vhngthnh/roberta-model), [BROS](https://www.kaggle.com/code/vhngthnh/bros-model), [PICK](https://www.kaggle.com/code/vhngthnh/pick-model).

## Kết quả test

| Model | Dataset | Epoch đã chạy | Best epoch val | Best val F1 | Test loss | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| roberta | sroie | 14 | 9 | 0.8702 | 0.0258 | 0.8682 | 0.8710 | 0.8696 |
| roberta | cord_v2 | 17 | 12 | 0.8614 | 0.2632 | 0.7651 | 0.8039 | 0.7761 |
| roberta | wildreceipt | 24 | 19 | 0.8573 | 0.4934 | 0.8684 | 0.8294 | 0.8319 |
| bros | sroie | 17 | 12 | 0.8951 | 0.0254 | 0.8848 | 0.8811 | 0.8829 |
| bros | cord_v2 | 28 | 23 | 0.8570 | 0.1839 | 0.7918 | 0.8221 | 0.7982 |
| bros | wildreceipt | 13 | 8 | 0.9062 | 0.2307 | 0.9238 | 0.9167 | 0.9185 |
| pick | sroie | 27 | 22 | 0.6035 | 0.6056 | 0.5568 | 0.6001 | 0.5619 |
| pick | cord_v2 | 24 | 19 | 0.4013 | 2.1337 | 0.3901 | 0.3783 | 0.3731 |
| pick | wildreceipt | 20 | 15 | 0.6225 | 4.6222 | 0.6290 | 0.6450 | 0.6308 |

Checkpoint được chọn theo F1 validation, sau đó nạp best_weights.pt để chấm test một lần trong mỗi train_model. Cả 9 lượt dừng sau 5 epoch không cải thiện, phù hợp patience=5. Các giá trị trong history hữu hạn. Đây là kiểm tra output và code đã lưu, không phải tái chạy hay kiểm tra nạp checkpoint trên máy local.

## Slide cần ghi gì

- Các số và đánh dấu BROS tốt nhất về Precision/Recall/F1 đều khớp các lượt này.
- Ghi rõ: “Kết quả trên tập test; Precision, Recall, F1 lấy trung bình theo trường (macro); đối sánh chính xác nhãn–giá trị sau chuẩn hóa chữ thường và khoảng trắng; seed 42.”
- F1 là trung bình F1 từng trường, không tính lại bằng trung bình Precision/Recall trong bảng.
- Không dùng Loss để xếp hạng giữa các kiến trúc: RoBERTa/BROS dùng token classification loss; PICK dùng CRF loss + 0.01 × graph loss. Có thể bỏ cột Loss khỏi bảng so sánh, hoặc giữ nhưng bỏ tô đậm.
- Các lượt vẫn dùng một hàm Macro-F1 nhãn–giá trị chung cho cả ba dataset; không gọi đây là đã chuyển sang protocol tham chiếu riêng từng dataset. Đặc biệt WildReceipt không chấm nhãn từng box trong các kết quả này.
- BROS cao nhất trong các lượt đã kiểm tra; chưa chứng minh luôn tốt nhất với mọi seed/cấu hình.

## Dữ liệu đầu vào và xử lý

Nguồn notebook gắn: [vhngthnh/reconia-data](https://www.kaggle.com/datasets/vhngthnh/reconia-data). get_data_dir trong bản đã chạy trỏ /kaggle/input/datasets/vhngthnh/reconia-data/<dataset>. Không gọi PaddleOCR; tên khóa ocr chỉ chứa text/box đọc từ annotation gốc.

| Dataset | Train | Val | Test | Cách chuẩn bị |
|---|---:|---:|---:|---|
| SROIE | 501 | 125 | 347 | Đọc img/box/key; chia 20% train gốc làm val, seed42; tách dòng thành từ và dấu hai chấm, nội suy box theo vị trí ký tự; gán BIO bằng khớp GT, có fallback gần đúng >0.85. |
| CORD-v2 | 800 | 100 | 100 | Giữ split có sẵn; đọc valid_line để lấy text, quad, category/is_key; đổi nhãn sang BIO; làm phẳng gt_parse thành trường–giá trị. |
| WildReceipt | 1014 | 253 | 472 | Đọc train.txt/test.txt và class_list.txt; chia 20% train làm val, seed42; các lớp _value được gán B-<class>, các lớp khác thành O. |

Ba bộ prepare chuyển box về [x1,y1,x2,y2] thang 0–1000 và lưu .cache/<dataset>/{train,val,test}.jsonl. Đây là biến đổi đầu vào cho model, không phải chạy OCR mới. Riêng box từ của SROIE là nội suy từ box dòng, không phải box từ được dataset cung cấp trực tiếp.

Đối chiếu 9 file JSONL của cả ba notebook: giống nhau từng byte giữa các lượt. Toàn bộ nội dung JSON, ID, split và GT cũng khớp 9 file prepared hiện có ở local. Không sửa dữ liệu.

| Model | Đầu vào thực tế | File xử lý trong bản đã chạy |
|---|---|---|
| RoBERTa | Text → tokenizer → input_ids, attention_mask; nhãn BIO ở subtoken đầu, phần còn lại -100; tối đa512token | models/roberta/dataset.py |
| BROS | Text như trên + bbox chuẩn hóa0–1 cho mỗi token; tối đa512token | models/bros/dataset.py |
| PICK | Ảnh resize512×512, text mã hóa ký tự, box theo ảnh resize, quan hệ hình học/độ dài; tối đa256box và64ký tự/box | models/pick/dataset.py |

Cả ba dự đoán nhãn rồi ghép text tương ứng thành trường–giá trị để utils/metrics.py đối chiếu GT. models/<model>/train.py điều khiển train, validation, lưu checkpoint và test.

## Mẫu dài bị cắt trong bản đã chạy

Đếm lại bằng tokenizer tải từ chính output; không chạy model và không tính lại test metric. Mỗi số là số tài liệu bị cắt do giới hạn đầu vào.

| Model | Dataset | Train | Val | Test |
|---|---|---:|---:|---:|
| roberta | sroie | 6 | 2 | 2 |
| roberta | cord_v2 | 0 | 0 | 0 |
| roberta | wildreceipt | 6 | 3 | 1 |
| bros | sroie | 2 | 1 | 1 |
| bros | cord_v2 | 0 | 0 | 0 |
| bros | wildreceipt | 6 | 3 | 1 |
| pick | sroie | 0 | 0 | 0 |
| pick | cord_v2 | 0 | 0 | 0 |
| pick | wildreceipt | 0 | 0 | 0 |

Ở test WildReceipt, RoBERTa bỏ hoàn toàn 31 entry có nhãn giá trị; BROS bỏ 3 entry có nhãn giá trị do cắt token. Chưa xác định mức thay đổi điểm nếu xử lý đủ các đoạn; không quy toàn bộ chênh lệch điểm cho hiện tượng này. PICK không mẫu nào vượt giới hạn box/ký tự ở 9 split.

## Vị trí code và sự khác nhau với local

Notebook đã chạy clone https://github.com/tamir39/reconia.git nhánh sub-dev-02 vào /kaggle/working/reconia. Sau clone notebook xóa .git, nên không có commit SHA được lưu trong phần output đã kiểm tra. Bản source .py thực tế vẫn nằm trong Output → reconia → prepare/models/utils.

Bản tải về để đối chiếu nằm tại D:/Projects/Reconia/.cache/thinh-notebook-review-20261024/<model>/output/reconia/. Các file kết quả nằm dưới results/<dataset>/<model>/test_metrics.json, history.csv, train.log.

Code đang chỉnh local thuộc nhánh codex/original-data-six-models. prepare/*.py và models/<model>/dataset.py vẫn là các điểm vào; riêng RoBERTa/BROS dùng chung utils/token_data.py, có chia cửa sổ512token/overlap64 để đọc hết tài liệu dài. Bản Kaggle đã chạy không dùng file chung này. Không lấy hành vi local mới để mô tả các số trên slide.

Các file local hiện tại:

- [Chuẩn bị SROIE](D:/Projects/Reconia/prepare/prepare_sroie.py)
- [Chuẩn bị CORD](D:/Projects/Reconia/prepare/prepare_cord_v2.py)
- [Chuẩn bị WildReceipt](D:/Projects/Reconia/prepare/prepare_wildreceipt.py)
- [Token hóa/chia đoạn RoBERTa và BROS](D:/Projects/Reconia/utils/token_data.py)
- [Đầu vào PICK](D:/Projects/Reconia/models/pick/dataset.py)
- [Hàm chấm điểm](D:/Projects/Reconia/utils/metrics.py)

Evidence máy đọc được: .cache/thinh-notebook-review-20261024/audit.json và slide-verification.json. Không tải checkpoint lớn, không sửa code train/notebook/data trong lần kiểm tra này.
