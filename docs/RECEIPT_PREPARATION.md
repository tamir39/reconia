# Chuẩn bị SROIE và CORD-v2 — 25/09/2026

Đã chuẩn bị dữ liệu cục bộ, hoàn tất và kiểm chứng GNN/RoBERTa trên Kaggle cho cả hai bộ.
Các lần chạy DocILE đã hoàn tất và được giữ riêng.

## Trạng thái Kaggle ngày 25/09

- Dataset private: [Reconia SROIE CORD Training Bundle](https://www.kaggle.com/datasets/phvngtngtm/reconia-sroie-cord-training-bundle), version 2 cho GNN; version 3 thêm scorer bốn key cho RoBERTa.
- [GNN SROIE](https://www.kaggle.com/code/phvngtngtm/reconia-gnn-sroie): smoke version 2 thành công; full run version 3 hoàn tất 20 epochs, best 15; 63 test đã kiểm chứng, entity F1 80,77%.
- [GNN CORD](https://www.kaggle.com/code/phvngtngtm/reconia-gnn-cord): smoke version 1 thành công; full run version 2 hoàn tất 20 epochs, best 19; 100 test đã kiểm chứng, entity F1 69,73%.
- Smoke dùng 16 train / 4 val, một epoch trên Tesla T4, nạp checkpoint ở tiến trình mới và xuất predictions. Full run học mới với seed 42, tối đa 20 epochs, patience 5; không resume checkpoint smoke.
- **122 tests passed** sau bổ sung scorer bốn key (117 ở lần chạy GNN). Bằng chứng và cấu hình smoke GNN: [receipts_kaggle_20260925.json](../reports/data/receipts_kaggle_20260925.json).

Upload trực tiếp bundle 2,7 GB qua mạng cục bộ quá chậm nên đã dừng trước khi tạo
dataset. Bundle private thay thế khoảng 2 MB gồm mã nguồn, manifest đối chiếu và
nhãn trường. `scripts/fetch_kaggle_receipts.py` tải nguồn public đã pin ngay trên
Kaggle. Mọi ảnh/OCR/key SROIE phải khớp byte; mọi ảnh và annotation CORD phải khớp
hash theo từng split và vị trí receipt. Arrow có thể khác cách tuần tự hóa giữa
máy, nên so nội dung ảnh/nhãn trước khi lưu checksum cho lần chạy.

Smoke SROIE lần đầu dừng trước training do JSON mapping sinh trên Windows dùng
CRLF, còn Linux dùng LF. Đã sửa riêng digest metadata mapping bằng JSON chuẩn hóa,
thêm regression test và chạy lại thành công. Không nới kiểm tra ảnh/OCR/key gốc.

Không đưa điểm smoke vào bảng benchmark. Kết quả GNN full run đã được tải về và
kiểm chứng; xem [bảng so sánh](../reports/verified_comparison.md). Notebook
[RoBERTa receipt](../notebooks/kaggle_roberta_receipts.ipynb) dùng cùng split và
backbone RoBERTa-base đã pin như DocILE, smoke 8 train / 4 val trước full run.

## RoBERTa receipt

[RoBERTa SROIE](https://www.kaggle.com/code/phvngtngtm/reconia-roberta-sroie) và
[RoBERTa CORD](https://www.kaggle.com/code/phvngtngtm/reconia-roberta-cord) đều private.
Smoke version 1 đã qua train/reload/inference trên 8 train / 4 val; SROIE chạy thêm
scorer bốn key thành công. Full version 2 đã hoàn tất cho cả hai bộ, seed 42,
tối đa 20 epochs, patience 5, batch 4 × accumulation 4. Backbone
`FacebookAI/roberta-base` pin revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`,
LR backbone 5e-5 / head 2e-4; xử lý toàn bộ cửa sổ OCR. Full run học mới, không
resume smoke. Best/latest và tokenizer được giữ trong output ZIP; không cần tải
ZIP smoke lớn để dùng cho benchmark. Bằng chứng:
[roberta_receipts_kaggle_20260925.json](../reports/data/roberta_receipts_kaggle_20260925.json).
ZIP full run đã tải và kiểm chứng toàn bộ CRC/hash, split và điểm field-value. SROIE: best 8/13 epochs, entity F1 88,37%, đoạn OCR F1 89,20%, bốn key gốc F1 73,60%. CORD: best 9/14 epochs, entity F1 91,99%, field-value F1 94,08%. Cả hai dừng sớm đúng patience 5. Xem [bằng chứng SROIE](../reports/runs/roberta_sroie_20260925/verification.json) và [CORD](../reports/runs/roberta_cord_20260925/verification.json).

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
không có ảnh giống hệt giữa các split trong từng dataset. **112 tests passed** sau rà soát lần hai khi
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
chọn `train`; full run kiểm tra đủ 63 hoặc 100 dự đoán test. Các bundle ảnh đầy đủ
vẫn lưu cục bộ; Kaggle sử dụng bundle nhỏ và tải nguồn trực tiếp như mô tả ở trên.

### Rà soát trước upload

Đã đối chiếu hash toàn bộ archive và từng file bên trong, đồng thời so byte của cả
626 cặp ảnh/OCR/key SROIE với release nguồn. Trình đóng gói từ chối file ngoài danh
sách cho phép và từ chối archive SROIE khác hash đã kiểm tra. Notebook kiểm tra hash
container, split SROIE, bỏ biến export cũ có thể ghi đè dataset, và yêu cầu thư mục
giải nén mới. Resume phải cùng dữ liệu, mã nguồn, seed và tổng epochs; không dùng
checkpoint smoke làm điểm bắt đầu cho benchmark đầy đủ.

Smoke test phát hiện và đã sửa lỗi suy luận CORD từ chối `--split val` vì nguồn
đặt tên `validation`. Pipeline chấp nhận alias hai chiều, vẫn từ chối split không
hợp lệ và lưu `source_split` thực tế trong metrics. Đã thêm regression test và
đóng gói lại mã nguồn trước khi chạy lại preflight.

Đã chạy các cell notebook từ bundle giải nén mới trên GPU RTX 3050 Ti cục bộ cho
cả hai dataset: 16 train / 4 val, một epoch, nạp lại checkpoint ở tiến trình mới,
xuất dự đoán và ZIP kết quả. Đường dẫn Kaggle được đổi sang cache cục bộ; bỏ bước
cài pip và hiển thị IPython (dùng dependency đã cài). Lần chạy GPU trong sandbox
bị Windows kết thúc; chạy ngoài sandbox thành công. Đây là **kiểm tra đường chạy**,
không phải kết quả benchmark, chưa xác minh môi trường Kaggle hoặc resume đầy đủ.
Xem [bằng chứng preflight](../reports/data/receipts_preflight_20260925.json).

## Sẵn sàng cho VLM và giới hạn đánh giá

Ảnh và nhãn đã được giữ lại cho Qwen2.5-VL và InternVL, hai họ mô hình Thịnh đề xuất.
Checkpoint đề xuất để thử bộ nhớ Kaggle: `Qwen/Qwen2.5-VL-3B-Instruct` và
`OpenGVLab/InternVL2_5-2B`; chưa tải trọng số, hiện thực adapter hoặc fine-tune.

`field-value-targets.json` của SROIE giữ các giá trị key gốc. Adapter BIO hiện tại
gán nhãn theo heuristic và có thể chia địa chỉ/công ty thành nhiều đoạn OCR.
Đã bổ sung `app/evaluation/sroie_values.py`: ghép đoạn dự đoán theo thứ tự OCR và
chấm với key gốc, dùng cùng schema cho VLM. GNN đạt F1 bốn key 54,70%, đã đối chiếu
trực tiếp với nhãn gốc. Điểm field-value từ các đoạn OCR vẫn tách riêng; xem
[quy tắc bốn key](EVALUATION.md#sroie-bốn-key-gốc-dùng-chung). CORD dùng transcript và box có chú
giải; cần ghi rõ đây là post-OCR parsing khi so với VLM đọc trực tiếp ảnh.
