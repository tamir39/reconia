# LayoutLMv3 — chuẩn bị thực nghiệm receipt

## Trạng thái ngày 25/09/2026

**126 tests passed.** Hai smoke version 1 trên Tesla T4 đã qua train, reload ở tiến
trình mới và suy luận đủ 4 val; hash predictions/mã nguồn, split và field-value
đã được kiểm tra cục bộ. SROIE đã qua cả scorer bốn key gốc.

Full version 2 đã hoàn tất và kiểm chứng trên [SROIE](https://www.kaggle.com/code/phvngtngtm/reconia-layoutlmv3-sroie)
và [CORD](https://www.kaggle.com/code/phvngtngtm/reconia-layoutlmv3-cord), học mới trên toàn bộ train.
Dataset private dùng version 4. Xem [bằng chứng và cấu hình](../reports/data/layoutlmv3_receipts_kaggle_20260925.json).
SROIE: best 9/14 epochs, test entity F1 94,50%, giá trị đoạn OCR F1 95,35%, bốn key gốc F1 75,90%. CORD: best 20/20 epochs, test entity F1 93,93%, field-value F1 95,51%. Đã kiểm tra ZIP/hash/split và tái tính field-value từ nhãn cục bộ; chưa chạy lại suy luận cục bộ. Xem [bảng so sánh đã kiểm chứng](../reports/verified_comparison.md).

## Phạm vi

Pipeline hỗ trợ SROIE, CORD-v2 và DocILE bằng ảnh trang, từ OCR và bbox có sẵn.
Dùng cùng split, vocabulary từ train, seed 42 và metric với GNN/RoBERTa đã kiểm chứng.
DocILE dùng [notebook riêng](../notebooks/kaggle_layoutlmv3_docile.ipynb) với
[bundle ảnh toàn trang đã đối chiếu OCR](DOCILE_IMAGES.md). DocILE smoke GPU đã qua kiểm chứng; full version 2 đã hoàn tất 20 epochs và kiểm chứng: test entity F1 83,19%, field-value F1 68,06%;
không dùng bundle OCR-only cho mô hình ảnh.

Backbone: `microsoft/layoutlmv3-base`, revision
`cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`. Theo
[tài liệu chính thức](https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/layoutlmv3),
processor dùng `apply_ocr=False` để nhận OCR/bbox của bộ dữ liệu.

## Đường chạy

- [Notebook Kaggle](../notebooks/kaggle_layoutlmv3_receipts.ipynb), mặc định `RUN_MODE = "smoke"`.
- Smoke: 8 train / 4 val, một epoch; nạp checkpoint ở tiến trình mới rồi chấm val.
- Full: SROIE 500/63/63, CORD 800/100/100, tối đa 20 epochs, patience 5 theo val entity F1.
- Batch 2 × accumulation 8, effective batch 16; backbone LR 5e-5, head LR 2e-4, weighted cross-entropy.
- Không dùng gradient checkpointing vì phiên bản LayoutLMv3 trong Transformers 4.57.6 không hỗ trợ tùy chọn đó.
- Phiên đầu chỉ hỗ trợ học mới. Không resume hoặc ghi đè thư mục checkpoint đã dùng.

Chạy cục bộ khi đã đặt `SROIE_DIR` hoặc `CORD_DIR` đến dữ liệu đầy đủ:

```bash
python -m app.training.train_layoutlmv3 --dataset sroie --epochs 20 --save-dir checkpoints/layoutlmv3-sroie --report-dir reports/layoutlmv3-sroie
python -m app.evaluation.predict --model layoutlmv3 --dataset sroie --checkpoint checkpoints/layoutlmv3-sroie/layoutlmv3_sroie_best.pt --split test
```

Runner `run_benchmark` hiện vẫn chỉ điều phối GNN/RoBERTa; dùng CLI riêng ở trên cho LayoutLMv3.

## Kiểm soát đầu vào và đầu ra

Mỗi trang được tokenize riêng thành các cửa sổ tối đa 512 token. Mỗi cửa sổ nhận
đúng ảnh của trang đó; mỗi từ được gán nhãn tại subword đầu tiên đúng một lần,
kể cả từ bị chia qua biên cửa sổ. Không bỏ phần cuối receipt. Policy:
`page-image-all-windows-first-subword-v1`.

Adapter receipt báo lỗi khi thiếu ảnh hoặc yêu cầu trang ngoài phạm vi, không tự
thay bằng ảnh trắng. Bộ kiểm thử kiểm tra nội dung ảnh tác động lên forward pass,
khớp trang/overflow, reload offline và một lượt train/test đầy đủ với model nhỏ.

Best/latest checkpoint giữ config, vocabulary, policy, optimizer/scheduler và lịch
sử; processor/tokenizer được lưu cùng để reload offline. Notebook giữ source hash,
data checksum, manifest đối chiếu ảnh/nhãn và thông tin backbone. Output giữ prediction,
checkpoint/split hashes, metric version, seed và thiết bị. SROIE xuất thêm điểm bốn key gốc.

Điểm smoke không phải kết quả benchmark. Full run chỉ được đưa vào
[bảng so sánh](../reports/verified_comparison.md) sau khi tải và kiểm chứng.
