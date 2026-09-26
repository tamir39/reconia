# LiLT — OCR và bố cục

LiLT dùng từ OCR và bbox có sẵn, không cần ảnh trang hoặc chạy OCR mới.
Checkpoint: `SCUT-DLVCLab/lilt-roberta-en-base`, revision
`96328faae824978e47bd0e8ea09f66b85efda5f5`.
[Model card](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) và
[tài liệu Transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/lilt.md).

Mỗi trang được chia thành cửa sổ 512 token riêng; mỗi từ chỉ có một first-subword
được giám sát. Tokenizer tương thích LayoutLMv3 đi cùng checkpoint được lưu để
nạp offline. Policy: `page-layout-all-windows-first-subword-v1`.

Cùng split và seed 42 với các baseline: DocILE 4.680/500/500,
SROIE 500/63/63, CORD 800/100/100. Vocabulary chỉ từ train. Receipt OCR export
được đối chiếu tất cả tài liệu và nhãn BIO với adapter gốc trước khi upload.
Giới hạn split/template DocILE và mapping SROIE vẫn áp dụng.

Fine-tune mới: tối đa 20 epochs, patience 5 theo validation entity F1,
batch 2 × accumulation 8, backbone LR 5e-5, head LR 2e-4.
Lưu best/latest checkpoint, tokenizer, cấu hình, history và dự đoán toàn bộ test.
Chưa hỗ trợ resume. Smoke dùng 8 train/4 val, 1 epoch và reload ở tiến trình mới.
Chỉ chạy full sau khi smoke qua; smoke không dùng để kết luận chất lượng.

```bash
python -m app.training.train_lilt --dataset docile --save-dir checkpoints/lilt-docile --report-dir reports/lilt-docile
python -m app.evaluation.predict --model lilt --dataset docile --checkpoint checkpoints/lilt-docile/lilt_docile_best.pt --split test
```

Chấm `ocr-bio-exact-span-v2` và `field-value-nfkc-casefold-v1` riêng biệt;
SROIE xuất thêm bốn key gốc. Đây không phải kết quả DocILE chính thức.
Chưa có full run LiLT được kiểm chứng; xem `reports/data/` để biết trạng thái Kaggle.
