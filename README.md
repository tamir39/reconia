# Reconia — Trích xuất thông tin hóa đơn

Repository phục vụ bài so sánh **RoBERTa, BROS, PICK, LayoutLMv3, Donut và Qwen2.5-VL** trên **SROIE, CORD-v2, WildReceipt**. Nhánh nộp source: [`main`](https://github.com/tamir39/reconia/tree/main).

Bản hiện tại có **source 6 model, 18 bộ kết quả test**, cùng **demo tương tác 5 model** và 30 trường minh họa đúng/sai. Qwen đã có đủ ba kết quả từ lịch sử Versions của Thịnh; inference Qwen chưa được tích hợp/kiểm chứng trong demo local.

## Bắt đầu trên Windows

Cần **Python 3.12 hoặc uv**, Internet khi cài dependency và quyền đọc repo. Tải nhánh `main` hoặc giải nén ZIP source. Nếu dùng Git:

```powershell
git clone --branch main --single-branch https://github.com/tamir39/reconia.git
cd reconia
```

1. Chạy `setup.bat` lần đầu. Script tạo/dùng `.venv`, bổ sung pip nếu thiếu và cài môi trường demo CPU.
2. Chạy `run.bat`. App mở tại <http://127.0.0.1:8766>; giữ cửa sổ lệnh mở trong lúc demo.
3. Chọn model/dataset để xem các ví dụ và điểm đã lưu. Để bấm **Chạy** trên ảnh test/ảnh ngoài cần thêm checkpoint tương ứng theo mục dưới.

Ảnh mẫu, OCR mẫu, nhãn, config và tokenizer nhỏ đã nằm trong repo. Xem ví dụ đã lưu không cần raw dataset hay checkpoint. Nút **Chạy** thực hiện inference thật bằng trọng số đã huấn luyện, không train lại.

## Lấy checkpoint để chạy thật

Trọng số lớn không nằm trong Git/ZIP source. Người nhận cần tài khoản Kaggle có quyền đọc các output của Thịnh, hoặc nhận bộ checkpoint tương ứng từ nhóm. Key được cấu hình riêng trên máy; không đặt credential vào repo.

Ví dụ lấy riêng ba checkpoint LayoutLMv3:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-assets.txt
.venv/Scripts/python.exe -m tools.fetch_assets --checkpoints --model layoutlmv3
```

Bỏ `--model layoutlmv3` để lấy đủ **15 checkpoint của 5 model trong demo**. Tool xác minh kích thước và SHA-256, chỉ đọc output, không gửi job train. Nếu output nguồn đã thay đổi và không khớp snapshot, tool dừng. Qwen không thuộc manifest tải checkpoint demo này.

Cấu trúc checkpoint:

```text
checkpoints/
└── <sroie|cord_v2|wildreceipt>/
    └── <roberta|bros|pick|layoutlmv3|donut>/
        └── best_weights.pt
```

Nếu đã giữ checkpoint ở nơi khác, đặt `RECONIA_CHECKPOINT_DIR` trỏ tới thư mục `checkpoints` đó trước khi mở app. Ví dụ PowerShell:

```powershell
$env:RECONIA_CHECKPOINT_DIR = "D:/ReconiaAssets/checkpoints"
./run.bat
```

Mỗi checkpoint được kiểm tra hash trước khi nạp. Lệnh `python -m tools.verify_final --weights` yêu cầu đủ 15 checkpoint; nếu chỉ tải một model, app vẫn chạy model đó và kiểm tra hash khi nạp.

## Demo và heatmap

Chọn model/dataset để xem hai hóa đơn test, mỗi thẻ minh họa một trường đúng hoặc sai. Các trường khác của cùng hóa đơn vẫn có thể sai. Có thể phóng ảnh, chạy lại dự đoán hoặc tải ảnh JPG/PNG/WebP của bạn lên.

Mẫu test dùng OCR chú thích theo quy trình đánh giá. Ảnh ngoài dùng RapidOCR cho RoBERTa/BROS/PICK/LayoutLMv3; Donut đọc ảnh trực tiếp. Model và dataset đang chọn xác định checkpoint dùng cho ảnh tải lên.

Heatmap hiện dùng **Occlusion cho một trường dự đoán**, hỗ trợ PICK, LayoutLMv3 và Donut. Vùng đỏ là vùng mà khi che đi làm điểm của đáp án đang xem giảm, không phải dấu hiệu đáp án đúng. Tín hiệu yếu được báo rõ. Chi tiết phương pháp và giới hạn: [demo/README.md](demo/README.md).

## Kết quả test

**Micro-F1 (%)**: cộng TP/FP/FN trên các trường và tài liệu; khớp giá trị sau chuẩn hóa chữ thường/khoảng trắng, giữ giá trị lặp.

| Model | SROIE | CORD-v2 | WildReceipt |
|---|---:|---:|---:|
| RoBERTa | 86.75 | 94.45 | 89.40 |
| BROS | 87.68 | 95.80 | 94.34 |
| PICK | 57.58 | 75.43 | 79.99 |
| LayoutLMv3 | 88.08 | 95.48 | 93.81 |
| Donut | 77.76* | 72.40* | 56.93* |
| Qwen2.5-VL | 78.46* | 81.92* | 74.46* |

**Ghi chú protocol:** Donut và Qwen trong source đã xuất dùng độ dài đáp án thật của batch để giới hạn số token sinh khi validation/test. Nội dung đáp án thật không được đưa vào prompt sinh, nhưng không mô tả protocol này là hoàn toàn độc lập nhãn chuẩn. Demo trên ảnh ngoài dùng giới hạn sinh cố định. Giữ nguyên điểm công bố; chưa chấm lại hai model với token budget độc lập nhãn.

Năm model đầu được đồng bộ từ output đã kiểm chứng ngày 21/11/2026; 30 ví dụ đã chạy lại bằng 15 checkpoint local. Qwen được đối chiếu source/log/history/cả 9 split ngày 24/11/2026, chưa nạp lại checkpoint để suy luận local. Các kết quả Qwen nằm ở **Versions của cùng notebook**:

| Dataset | Output version | Best validation epoch | Early stop epoch |
|---|---:|---:|---:|
| SROIE | 1 | 4 | 9 |
| CORD-v2 | 3 | 5 | 10 |
| WildReceipt | 5 | 4 | 9 |

Nguồn Kaggle: [RoBERTa](https://www.kaggle.com/code/vhngthnh/roberta-model), [BROS](https://www.kaggle.com/code/vhngthnh/bros-model), [PICK](https://www.kaggle.com/code/vhngthnh/pick-model), [LayoutLMv3](https://www.kaggle.com/code/vhngthnh/layoutlmv3-model), [Donut](https://www.kaggle.com/code/vhngthnh/donut-model), [Qwen](https://www.kaggle.com/code/vhngthnh/qwen-vl-model).

Giá trị đầy đủ, history/log: [reports/benchmarks/](reports/benchmarks/). Nguồn và hash source: [provenance.json](reports/benchmarks/provenance.json). Version từng kết quả Qwen: [additional-models.json](reports/benchmarks/additional-models.json); [kiểm chứng Qwen](reports/verification-qwen-20261124.json). Không lấy số từ báo cáo Macro-F1 cũ.

## Cây thư mục trong Git/ZIP source

```text
reconia/
├── models/                 Source 6 model; Qwen có README/config riêng
├── notebooks/              6 notebook gọn, mọi cờ chạy mặc định tắt
├── prepare/                Chuyển raw data, manifest kiểm tra 9 split
├── utils/                  Hàm chung, Micro-F1, biểu đồ
├── demo/                   App, 30 ví dụ, 16 ảnh, nhãn/config/tokenizer
├── reports/
│   ├── benchmarks/         18 bộ metrics/history/log và provenance
│   ├── archive/            Báo cáo, ghi chú, bản thảo lịch sử
│   └── verification-*.json Bằng chứng kiểm tra source/demo/kết quả
├── tests/                  Kiểm tra scorer, dữ liệu, notebook, heatmap, Qwen
├── tools/                  Tải tài nguyên, kiểm chứng, so sánh, xuất ZIP
├── archive/                Source cũ để đối chiếu lịch sử
├── README.md               Hướng dẫn hiện tại
├── SUBMISSION.md           Phạm vi và cách bàn giao bản nộp
├── setup.bat               Cài môi trường demo CPU
├── run.bat                 Mở demo
├── requirements.txt        Dependency demo đã kiểm tra trên Python 3.12
├── requirements-assets.txt Dependency tải tài nguyên Kaggle, tùy chọn
├── requirements-qwen.txt   Dependency Qwen GPU riêng, chưa khóa runtime
├── pytest.ini              Chỉ chạy tests hiện tại
├── .gitattributes          Quy tắc file text/binary
└── .gitignore              Loại tài nguyên local khỏi Git
```

Trên máy làm việc còn có `.venv/`, `.cache/`, `data/`, `weights/`, `checkpoints/`, `results/`, `output/`, `dist/` và `.env`. Chúng không thuộc source được nộp. Không nén nguyên thư mục làm việc; dùng ZIP do tool xuất để tránh kèm credential, môi trường hoặc dữ liệu lớn.

## Train tham khảo

Không cần train để mở demo hay xem kết quả. Khi chủ động làm thí nghiệm mới, dùng checkout riêng để giữ checkpoint đã chốt và môi trường GPU phù hợp.

| Model | Đầu vào | Batch × tích lũy | LR |
|---|---|---:|---:|
| RoBERTa | Chữ OCR | 8 × 1 | 3e-5 |
| BROS | Chữ + box OCR | 8 × 1 | 3e-5 |
| PICK | Ảnh + chữ + box OCR | 4 × 2 | 5e-4 |
| LayoutLMv3 | Ảnh + chữ + box OCR | 4 × 2 | 3e-5 |
| Donut | Ảnh + task token theo dataset | 4 × 2 | 3e-5 |
| Qwen2.5-VL | Ảnh + system/user prompt liệt kê trường | 2 × 4 | 1e-4 |

Default source đã xuất: tối đa 25 epoch, patience 5, seed 42; chọn best validation F1 rồi chấm test. Trainer dừng khi checkpoint đã tồn tại; `--resume` chỉ dùng cho lượt tương thích. Qwen dùng QLoRA 4-bit/CUDA và môi trường riêng theo [models/qwen_vl/README.md](models/qwen_vl/README.md); dependency này không được cài bởi `setup.bat`.

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| SROIE | 501 | 125 | 347 |
| CORD-v2 | 800 | 100 | 100 |
| WildReceipt | 1014 | 253 | 472 |

Raw data: `data/raw/<dataset>/`, hoặc `RECONIA_DATA_DIR` trỏ tới thư mục chứa ba dataset. Lấy split đã chốt bằng `python -m tools.fetch_assets --prepared`, hoặc chuyển raw bằng module cụ thể: `prepare.prepare_sroie`, `prepare.prepare_cord_v2`, `prepare.prepare_wildreceipt`. Tải JSONL không tải ảnh raw phục vụ train.

Trước train, chạy `python -m tools.verify_final --prepared` để kiểm tra số hàng, thứ tự, OCR, box và nhãn của 9 split. Notebook cũng kiểm tra khi bật train. Năm notebook đầu mặc định chọn `all`; Qwen mặc định chọn `wildreceipt`. Mọi cờ chạy đều tắt khi mở; `DATASET="all"` vẫn chịu giới hạn thời gian Kaggle.

Giữ split đã báo cáo. CORD-v2 có ghi nhận một số mẫu trùng nội dung giữa các split trong kiểm toán trước, nên không mô tả test hoàn toàn độc lập. Không lọc trùng rồi gắn lại điểm cũ.

## Kiểm tra và xuất bản nộp

Chạy tại thư mục gốc sau khi cài môi trường:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m tools.verify_final
.venv/Scripts/python.exe -m tools.compare
```

Các lệnh trên không cần checkpoint và không train. Đã có đủ 15 checkpoint thì có thể kiểm tra toàn bộ hash và chạy lại 30 ví dụ:

```powershell
.venv/Scripts/python.exe -m tools.verify_final --weights
.venv/Scripts/python.exe -m tools.check_demo_predictions
```

Từ **Git checkout**, sau khi commit source cần nộp:

```powershell
.venv/Scripts/python.exe -m tools.export_submission
```

ZIP tại `dist/reconia-final-source.zip`, kèm `SOURCE_MANIFEST.json` ghi commit và hash file. Giải nén ZIP để mở app không cần Git; tạo lại ZIP bằng tool yêu cầu Git checkout. Source ZIP không chứa trọng số/raw data. Xem [SUBMISSION.md](SUBMISSION.md) để bàn giao đúng phạm vi. Bản thảo học thuật ở `reports/archive/` chưa được coi là báo cáo final đã duyệt.
