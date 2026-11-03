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
Các lượt RoBERTa/BROS đã hoàn thành lấy tối đa 512 token đầu mỗi tài liệu.
LayoutLMv3 mới dùng cùng chính sách một đoạn tối đa 512 token; không chia cửa sổ.
Code RoBERTa/BROS local có thử nghiệm cửa sổ 512/overlap 64, nhưng không phải source
đã tạo ra chín kết quả cũ; không dùng lại nhánh thử nghiệm đó để mô tả kết quả đã chạy.

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

Bản raw đã đối chiếu trên máy này nằm tại `D:/Projects/Reconia/.cache/preprocessing-complete/raw`.
Đặt `RECONIA_DATA_DIR` tới thư mục đó khi chạy local.

Trong mỗi notebook, chọn `DATASET = "sroie"`, `"cord_v2"` hoặc `"wildreceipt"`.
Phần **Prepare and inspect** in đường dẫn dữ liệu và hiển thị một mẫu train:
ảnh, năm mục text/box đầu tiên và GT. Đây là bản xem dữ liệu trước khi adapter tạo tensor;
`target_gt` là đáp án, không phải đầu vào suy luận.

Trong Data Explorer của Kaggle, các đường dẫn sau tính từ gốc dataset:

| Bộ | Ảnh mẫu | Text/box và GT gốc |
|---|---|---|
| SROIE | `sroie/train/img/X51006387847.jpg` | Cùng tên trong `sroie/train/box/` là text/box; `sroie/train/key/` là GT, đuôi `.txt` |
| CORD-v2 | `cord_v2/images/train/0.png` | Bản ghi đầu của `cord_v2/train/`: cột `ground_truth`, `valid_line` chứa text/box/nhãn, `gt_parse` chứa đáp án |
| WildReceipt | `wildreceipt/image_files/Image_23/13/55f624b505e49e4bec8993810feca52465a07449.jpeg` | Tìm `file_name` này trong `wildreceipt/train.txt`; `annotations` chứa text/box/nhãn, tên nhãn ở `class_list.txt` |

CORD lưu annotation trong Arrow, nên notebook đọc được bằng `datasets.load_from_disk`;
Data Explorer có thể không hiển thị từng bản ghi như file JSON/TXT.

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

LayoutLMv3 dùng cùng cấu trúc trainer với RoBERTa/BROS/PICK:
`train_epoch`, `evaluate_epoch`, `evaluate_test`, `save_history`, `train_model`, `parse_args`.
Dataset/collator đặt tên `LayoutLMv3Dataset` và `LayoutLMv3DataCollator`.
Phần riêng là `pixel_values` từ ảnh và box nguyên ở thang 0–1000;
processor tắt OCR để dùng text/box gốc. Batch 4 × tích lũy 2, LR 3e-5.
Checkpoint gồm `best_weights.pt` và `last_checkpoint.pt`; kết quả gồm
`train.log`, `history.csv`, biểu đồ và `test_metrics.json` khi chạy test riêng.
Checkpoint resume của trainer LayoutLMv3 cũ dùng `utils.training.fit` có cấu trúc khác;
giữ run cũ riêng và bắt đầu run mới khi chuyển sang trainer này.

Donut cũng dùng nhóm hàm train/evaluate/test, tên biến và định dạng kết quả như các model trên;
dataset/collator là `DonutDataset` và `DonutDataCollator`.
Donut nhận ảnh 1280 × 960, không đưa text/box gốc vào model, và học sinh JSON
`{"field": ["value", ...]}` thay vì nhãn BIO. Batch 1 × tích lũy 8, LR 3e-5.
Validation loss dùng target JSON; Macro-F1 chấm JSON được sinh riêng từ ảnh và BOS,
không dùng target làm đầu vào sinh. JSON sai định dạng được tính là dự đoán rỗng;
giữ nguyên giá trị lặp. Target dài hơn giới hạn báo lỗi, không cắt âm thầm.
`--max_new_tokens` mặc định 1535, phải nằm trong giới hạn vị trí của decoder sau BOS.
Checkpoint resume từ trainer Donut cũ dùng `utils.training.fit` có cấu trúc khác;
giữ run cũ riêng khi chuyển sang trainer này.

Qwen dùng cùng nhóm hàm trainer và các bước notebook, với `QwenDataset` và `QwenDataCollator`.
Model `Qwen/Qwen2.5-VL-3B-Instruct` nhận ảnh và prompt cố định chứa tên các trường;
không đưa giá trị GT hoặc text/box annotation vào prompt. Target là JSON như Donut.
Chỉ học LoRA trên `q_proj`, `k_proj`, `v_proj`, `o_proj` của phần ngôn ngữ;
đóng băng model gốc và nhánh ảnh. LoRA r=16, alpha=32, dropout=0.05;
batch 1 × tích lũy 8, LR 1e-4. Đây là LoRA, chưa dùng lượng tử hóa QLoRA.
Loss bỏ qua toàn bộ prompt/ảnh và padding, chỉ học câu trả lời JSON và EOS.
Training padding bên phải; sinh chuỗi padding bên trái và chỉ chấm phần sinh mới sau prompt.
Giới hạn ảnh 256–768 visual token, đầu ra mặc định 2048 token;
tổng prompt/ảnh + ngân sách đầu ra không vượt 8192 hoặc giới hạn model nếu nhỏ hơn.
Target quá dài báo lỗi; JSON lỗi chấm là dự đoán rỗng, không sửa bằng GT.
Tên checkpoint/log giống các model trước, nhưng `best_weights.pt` và
`model_state_dict` trong `last_checkpoint.pt` chỉ chứa adapter LoRA;
`adapter_config.json` lưu cấu hình adapter. Cần giữ đúng base weights trong `weights/qwen_vl`
để resume hoặc test. Checkpoint resume từ trainer Qwen cũ dùng `utils.training.fit`
có cấu trúc khác; giữ run cũ riêng khi chuyển sang trainer này.

## Kiểm tra code

```bash
python -m unittest discover -s tests -v
```

Kiểm tra local chưa thay thế chạy thử GPU với trọng số đầy đủ. Xem
[biên bản kiểm tra](reports/original-data-integration-20261023.md).
Test LayoutLMv3 dùng tokenizer/processor và model nhỏ khởi tạo cục bộ,
kiểm tra ảnh/box, cắt đầu vào 512 token nhưng giữ GT đầy đủ, train/validation, resume
và test riêng mà không tải weights.
Test Donut dùng encoder DonutSwin và decoder mBART nhỏ khởi tạo cục bộ để kiểm tra
train, sinh chuỗi, resume, test riêng, dịch target, padding và giới hạn độ dài.
Kiểm tra thêm việc sinh không nhận GT, JSON lỗi và giá trị lặp khi chấm điểm.
Test Qwen dùng Qwen2.5-VL nhỏ và LoRA thật, khởi tạo cục bộ; kiểm tra mask loss,
padding, ngân sách context, đóng băng base model, checkpoint adapter, resume và test riêng.

## Đối chiếu sáu model với bài báo (27/10/2026)

Cả sáu notebook có cùng sáu mục; trainer cùng nhóm hàm và cùng cách chấm
Macro Precision/Recall/F1 theo trường–giá trị. Collator đặt tên
`RobertaDataCollator`, `BrosDataCollator`, `PickDataCollator`,
`LayoutLMv3DataCollator`, `DonutDataCollator`, `QwenDataCollator`.
Đây là thí nghiệm so sánh các cấu hình của nhóm, không phải tái lập toàn bộ protocol từng bài.

| Model và bài nguồn | Cách dùng trong repo / khác biệt cần ghi rõ |
|---|---|
| [RoBERTa — Liu et al., 2019](https://arxiv.org/abs/1907.11692) | Nạp `roberta-base`, thêm đầu BIO; không thực hiện lại tiền huấn luyện. Bài gốc không đề xuất pipeline hóa đơn này. |
| [BROS — Hong et al., 2022](https://arxiv.org/abs/2108.04539) | Nạp BROS, thêm đầu token classification; không triển khai entity linking/SPADE. SROIE* trong bài gốc là bản sửa, không cùng split SROIE hiện tại. |
| [PICK — Yu et al., 2020](https://arxiv.org/abs/2004.07464) | Bản triển khai giản lược: BiLSTM/CRF riêng từng box, graph loss dùng khoảng cách L2 chưa bình phương và chặn số mũ; không dùng lexicon SROIE. Kết quả chỉ đại diện bản này. LR mới đồng nhất 1e-4, khác cách chia LR của lượt cũ trong report giữa kỳ. |
| [LayoutLMv3 — Huang et al., 2022](https://arxiv.org/abs/2204.08387) | Nạp base, fine-tune BIO với text/box annotation và ảnh. Không thực hiện lại các mục tiêu tiền huấn luyện MLM/MIM/WPA. |
| [Donut — Kim et al., 2022](https://arxiv.org/abs/2111.15664) | Giữ encoder–decoder từ ảnh; dùng JSON thuần và BOS để thống nhất output. Bài gốc dùng task prompt và special token theo trường để chuyển thành JSON. |
| [Qwen2.5-VL — Bai et al., 2025](https://arxiv.org/abs/2502.13923) | Dùng bản 3B-Instruct và LoRA cho tác vụ hóa đơn; không tái lập pretraining/post-training của technical report. Ngân sách ảnh/prompt là cấu hình của nhóm. |

Text/box annotation được cung cấp cho RoBERTa/BROS/PICK/LayoutLMv3, còn Donut/Qwen
nhận ảnh và phải tự đọc chữ. Không dùng loss để xếp hạng giữa các kiến trúc.
Ba model cũ đã chạy xong; tiếp tục train LayoutLMv3/Donut/Qwen theo cùng dữ liệu và
hàm chấm đã đối chiếu với source cũ. Không thay checkpoint/kết quả cũ bằng hành vi
RoBERTa/BROS local mới dùng cửa sổ hoặc cấu hình PICK local đã đổi.

Kiểm tra CPU gồm forward/backward/evaluate bằng RoBERTa, BROS, LayoutLMv3,
Donut và Qwen kích thước nhỏ; PICK dùng các module thật thu nhỏ, backbone ResNet-50
khởi tạo ngẫu nhiên để kiểm tra offline. Các kiểm tra này không đo chất lượng pretrained
hoặc khả năng chứa batch thật trên GPU. Trước full train trên Kaggle cần dùng đúng source mới,
gắn dataset, tải weights/cache ResNet và chạy thử ít nhất một batch train + validation/generate
với cấu hình thật. Notebook không tự pull checkout đã tồn tại.

## Mốc dữ liệu và cách chấm của ba lượt đã hoàn thành

Mốc là source thực tế trong Output Kaggle của RoBERTa/BROS/PICK ngày 24/10/2026,
không phải bản local đã chỉnh sau đó. Đã đối chiếu cả ba bản source và 9 split prepared:
nội dung JSON, thứ tự mẫu, ID, text/box, BIO và GT trùng nhau; khác xuống dòng không
được xem là thay đổi dữ liệu. Khác thư mục giải nén ảnh CORD được chuẩn hóa theo suffix đường dẫn.

`prepare/completed_baseline_manifest.json` ghi hash nguồn tham chiếu, hash nội dung
từng split và danh mục nhãn theo đúng hàm export của lượt cũ: SROIE 9, CORD-v2 49,
WildReceipt 13 nhãn, gồm O. Không tự thêm các nhãn I- chưa xuất hiện như bộ nhãn local mở rộng.
LayoutLMv3 dùng danh mục này cho đầu BIO; Qwen dùng tên trường từ danh mục này cho prompt.
Donut không có đầu BIO, dùng trực tiếp GT của cùng các bản ghi.

Cả ba trainer mới và notebook xác minh prepared data với manifest trước khi tải model/train.
Khác split, thứ tự, annotation hoặc GT sẽ báo lỗi. Không sửa dữ liệu để vượt qua kiểm tra;
đối chiếu lại gói raw và bước prepare. Kiểm tra này khóa annotation/GT và đường dẫn tương đối,
không thay thế kiểm chứng hash nội dung ảnh raw.

Hàm chấm được đối chiếu bằng fixture lấy từ scorer của source đã chạy: khớp chính xác
trường–giá trị sau lower/gộp khoảng trắng, ghép giá trị lặp một-một, macro theo trường
trên toàn tập; không làm tròn/chuyển số tiền. Bản hiện tại chỉ thêm chặn sai số lượng
dự đoán/đầu vào rỗng. Không cắt GT theo độ dài input của model.

Checkpoint và `test_metrics.json` của ba model mới ghi
`data_protocol=completed-baselines-20261024`; resume từ checkpoint khác protocol bị chặn.
Các bài toán ảnh → JSON của Donut/Qwen vẫn có input riêng, nhưng chấm với cùng GT/scorer.

Đã chấm lại offline 2.757 dự đoán lưu từ 9 lượt cũ: scorer hiện tại và scorer source cũ
cho cùng kết quả; cả 9 bộ điểm khớp kết quả đã lưu ở 4 chữ số thập phân. Không chạy lại
inference/train và không thay kết quả cũ. Bằng chứng nằm trong
`reports/completed-baseline-alignment-20261027.json`.
