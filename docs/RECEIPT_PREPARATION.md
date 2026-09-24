# Chuẩn bị SROIE và CORD-v2 — 25/09/2026

Chuẩn bị dữ liệu cục bộ trong lúc RoBERTa/DocILE chạy trên Kaggle. Không khởi chạy
thí nghiệm mới hoặc thay đổi notebook/dataset DocILE đang dùng.

## Nguồn và split

| Dữ liệu | Nguồn | Train / val / test |
|---|---|---|
| SROIE | [Release v2.0 do Thịnh giới thiệu](https://github.com/NjoyimPeguy/ICDAR-2019-RRC-SROIE/releases/tag/v2.0) | 500 / 63 / 63 nội bộ |
| CORD-v2 | [NAVER CLOVA](https://huggingface.co/datasets/naver-clova-ix/cord-v2), revision `7f0115a4b758a71d6473b8d085751692da2fef98` | 800 / 100 / 100 chính thức |

SROIE dùng 626 ảnh train gốc có đủ file ảnh, OCR và key annotation. Release chứa cả
bản sao có hậu tố `(1)`, `(2)`; chỉ chọn tên chuẩn `X<digits>`. Không dùng ảnh test
chính thức để train. Các ID nguồn được sắp xếp và ánh xạ cố định sang `000`–`625`;
giữ các vị trí trong `app/data/sroie_splits.json`. **Chưa có mapping gốc của Thịnh,
nên không khẳng định thành viên split giống thí nghiệm cũ.** Mapping mới, SHA-256
của archive và split được lưu trong `source-id-map.json` và đi cùng bundle.

Receipt `X51005663280` thiếu key `address` trong nguồn. Giữ nguyên nhãn hiện có,
ghi số nhãn thiếu trong báo cáo; không tự bổ sung hay loại receipt khỏi split.

## Tạo lại và kiểm tra

Từ thư mục gốc, dùng Python của môi trường dự án:

```powershell
python -m scripts.prepare_receipts --download --bundle
```

Đầu ra mặc định: `.cache/receipt-prep/` (được Git bỏ qua):

- `sroie/`: ảnh JPEG, OCR CSV, key JSON, mapping và split.
- `cord/`: Hugging Face DatasetDict với ảnh và ground truth.
- `validation.json`: kiểm tra mọi receipt, số lượng split, ảnh trùng giữa split,
  tọa độ chuẩn hóa, số từ/trường và khả năng tạo nhãn BIO.
- `kaggle/`: hai container ZIP đuôi `.bin`, mã nguồn allowlist, hash manifest và báo cáo.

CORD được đọc bằng `datasets==5.0.1`, `pyarrow==23.0.1`. PyArrow 25.0.1 trên máy
Windows hiện tại bị Application Control chặn DLL; bản 23.0.1 đã nạp thành công.
Các archive không chứa API key, `.env` hoặc trọng số mô hình.

Kết quả kiểm tra ngày 25/09: **1.626/1.626 receipt** đọc được ảnh, OCR và nhãn;
không có ảnh giống hệt giữa các split trong từng dataset. **107 tests passed** khi
cấu hình đủ dữ liệu cục bộ. Bằng chứng: [validation report](../reports/data/receipts_20260925.json).
Bundle SROIE khoảng 357 MB, CORD khoảng 2,32 GB; dữ liệu ảnh chiếm phần lớn dung lượng.

Để chạy kiểm tra adapter cục bộ:

```powershell
$env:SROIE_DIR = (Resolve-Path .cache/receipt-prep/sroie).Path
$env:CORD_DIR = (Resolve-Path .cache/receipt-prep/cord).Path
python -m pytest tests/ -q
```

Notebook [kaggle_gnn_receipts.ipynb](../notebooks/kaggle_gnn_receipts.ipynb) chuẩn bị
đường chạy GNN cho từng dataset, mặc định `RUN_MODE="smoke"`. Attach bundle receipt
riêng, chọn `DATASET`, bật GPU và Internet. Notebook kiểm tra hash dữ liệu/mã nguồn,
chạy smoke trên validation và nạp lại checkpoint. Chỉ sau khi smoke thành công mới
chọn `train`; full run kiểm tra đủ 63 hoặc 100 dự đoán test. Notebook chưa được chạy
trên Kaggle trong bước chuẩn bị này. Bundle hiện lưu cục bộ, chưa upload.

## Sẵn sàng cho VLM và giới hạn đánh giá

Ảnh và nhãn đã được giữ lại cho Qwen2.5-VL và InternVL, hai họ mô hình Thịnh đề xuất.
Checkpoint đề xuất để thử bộ nhớ Kaggle: `Qwen/Qwen2.5-VL-3B-Instruct` và
`OpenGVLab/InternVL2_5-2B`; chưa tải trọng số, hiện thực adapter hoặc fine-tune.

`field-value-targets.json` của SROIE giữ các giá trị key gốc. Adapter BIO hiện tại
gán nhãn theo heuristic và có thể chia địa chỉ/công ty thành nhiều đoạn OCR.
**Trước khi so VLM với GNN/RoBERTa trên SROIE phải nối dự đoán theo trường và tích hợp
chấm với key gốc cho tất cả mô hình.** Điểm field-value hiện tại từ các đoạn OCR
không tự động tương đương điểm bốn trường này. CORD dùng transcript và box có chú
giải; cần ghi rõ đây là post-OCR parsing khi so với VLM đọc trực tiếp ảnh.
