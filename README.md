# Reconia

Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin từ hóa đơn.

Source huấn luyện của 6 mô hình lấy từ nhánh [dev](https://github.com/tamir39/reconia/tree/dev) tại commit `3aed933`. Nhánh `main` bổ sung demo và cách chạy local. Các file trong `models/`, `prepare/`, `notebooks/`, `utils/`, hai tool gốc và `requirements.txt` được giữ như `dev`.

## Chạy demo trên Windows

Cần Python 3.12 hoặc `uv`.

1. Chạy `setup.bat` lần đầu để cài môi trường demo CPU.
2. Chạy `run.bat`, mở app tại <http://127.0.0.1:8766>.
3. Chọn model và dataset. Hai thẻ minh họa một trường đúng và một trường sai trên tập test. Bấm **Chạy** để suy luận lại hoặc tải ảnh của bạn lên.

Demo hỗ trợ **RoBERTa, BROS, PICK, LayoutLMv3 và Donut** trên SROIE, CORD-v2, WildReceipt. Qwen có source huấn luyện nhưng chưa được tích hợp suy luận trong demo. Có thể xem các ví dụ đã lưu mà chưa có checkpoint; chạy suy luận thật cần checkpoint tương ứng.

Trọng số đặt tại:

```text
checkpoints/<dataset>/<model>/best_weights.pt
```

Nếu đã lưu ở nơi khác, đặt `RECONIA_CHECKPOINT_DIR` trỏ tới thư mục `checkpoints` trước khi mở app. Để tải checkpoint đã huấn luyện, cần tài khoản Kaggle có quyền đọc output của nhóm:

```powershell
.venv/Scripts/python.exe -m pip install -r demo/requirements-assets.txt
.venv/Scripts/python.exe -m demo.fetch_assets --model layoutlmv3
```

Bỏ `--model` để tải đủ 15 checkpoint. Lệnh chỉ tải output, không chạy huấn luyện. Demo dùng SHA-256 để nhận diện đúng file trọng số của các ví dụ đã lưu. Checkpoint, raw data và credential không nằm trong Git.

Mẫu test dùng OCR chú thích. Ảnh ngoài dùng RapidOCR cho bốn mô hình nhận chữ/hộp; Donut đọc ảnh trực tiếp. Model và dataset đang chọn quyết định checkpoint và bộ trường đầu ra. Ảnh hỗ trợ JPG, PNG, WebP tối đa 12 MB.

Heatmap dùng **che vùng (Occlusion)** cho một trường dự đoán, có trên PICK, LayoutLMv3 và Donut. Vùng đỏ là vùng mà khi che đi làm điểm của đáp án đang xem giảm; nó không đánh dấu đáp án đúng. Với PICK/LayoutLMv3 có thể chọn che riêng ảnh hoặc ảnh cùng chữ OCR.

## Huấn luyện

`requirements.txt` và các notebook là môi trường/quy trình huấn luyện từ `dev`, tách với môi trường demo trong `demo/requirements.txt`. Notebook có các lệnh thực nghiệm; chỉ thực thi khi muốn chuẩn bị dữ liệu hoặc train.

Dữ liệu gốc: [vhngthnh/reconia-data](https://www.kaggle.com/datasets/vhngthnh/reconia-data). Source tìm dataset tại các đường dẫn trong `utils/common.py`, gồm mount Kaggle và `<thư mục chạy>/<dataset>`.

```bash
pip install -r requirements.txt
python -m prepare.prepare_sroie
python -m prepare.prepare_cord_v2
python -m prepare.prepare_wildreceipt
python -m tools.download_weights --model roberta
python -m models.roberta.train --dataset sroie
```

JSONL tại `.cache/<dataset>/`; pretrained weights tại `weights/<model>/`; checkpoint tại `checkpoints/<dataset>/<model>/`; metrics, history và log tại `results/<dataset>/<model>/`. `python -m tools.compare` tổng hợp các kết quả có trong `results/`.

| Model | Đầu vào | Batch × tích lũy | Learning rate |
|---|---|---:|---:|
| RoBERTa | Chữ OCR | 8 × 1 | 3e-5 |
| BROS | Chữ và hộp OCR | 8 × 1 | 3e-5 |
| PICK | Ảnh, chữ và hộp OCR | 4 × 2 | 5e-4 |
| LayoutLMv3 | Ảnh, chữ và hộp OCR | 4 × 2 | 3e-5 |
| Donut | Ảnh và task token | 4 × 2 | 3e-5 |
| Qwen2.5-VL | Ảnh và chỉ dẫn | 2 × 4 | 1e-4 |

Mặc định tối đa 25 epoch, patience 5, seed 42, chọn checkpoint theo validation F1 rồi chấm test. `--resume` bật lưu và tiếp tục trạng thái huấn luyện. Qwen cần môi trường CUDA phù hợp để dùng QLoRA 4-bit.

## Kết quả đã lưu trên Kaggle

Test **Micro-F1 (%)**, khớp trường–giá trị sau chuẩn hóa chữ thường và khoảng trắng, giữ giá trị lặp.

| Model | SROIE | CORD-v2 | WildReceipt |
|---|---:|---:|---:|
| RoBERTa | 86.75 | 94.45 | 89.40 |
| BROS | 87.68 | 95.80 | 94.34 |
| PICK | 57.58 | 75.43 | 79.99 |
| LayoutLMv3 | 88.08 | 95.48 | 93.81 |
| Donut | 77.76 | 72.40 | 56.93 |
| Qwen2.5-VL | 78.46 | 81.92 | 74.46 |

Nguồn: [RoBERTa](https://www.kaggle.com/code/vhngthnh/roberta-model), [BROS](https://www.kaggle.com/code/vhngthnh/bros-model), [PICK](https://www.kaggle.com/code/vhngthnh/pick-model), [LayoutLMv3](https://www.kaggle.com/code/vhngthnh/layoutlmv3-model), [Donut](https://www.kaggle.com/code/vhngthnh/donut-model), [Qwen](https://www.kaggle.com/code/vhngthnh/qwen-vl-model). Qwen: SROIE ở version 1, CORD-v2 ở version 3, WildReceipt ở version 5.

Donut và Qwen dùng độ dài nhãn chuẩn để đặt giới hạn sinh khi đánh giá; demo ảnh ngoài dùng giới hạn cố định. CORD-v2 có một cặp train–test và hai cặp train–validation trùng OCR/nhãn. Các điểm trên giữ theo split và protocol của output gốc.

## Cấu trúc source

```text
models/          6 model từ dev
notebooks/       6 notebook huấn luyện từ dev
prepare/         Chuyển dữ liệu gốc sang JSONL
utils/           Hàm chung, metrics, biểu đồ và visualization từ dev
tools/           Tải pretrained weights và tổng hợp kết quả từ dev
demo/            App, ảnh/OCR mẫu, kết quả minh họa, config và tokenizer
requirements.txt Dependency huấn luyện từ dev
setup.bat        Cài môi trường demo
run.bat          Mở demo
```

Nhánh `dev` tiếp tục giữ code phát triển của Thịnh. Báo cáo học thuật, bản nháp, kiểm tra tự động và tài liệu làm việc được quản lý riêng, không nằm trong source này.
