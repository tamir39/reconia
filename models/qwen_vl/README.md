# Qwen của Thịnh

Nguồn: https://www.kaggle.com/code/vhngthnh/qwen-vl-model — version 5, kiểm tra ngày 24/11/2026. Base model `Qwen/Qwen2.5-VL-3B-Instruct`, QLoRA NF4 4-bit, r=16, alpha=32, dropout=0.05; target các projection attention/MLP của language model. Image budget 256–512 visual tokens. Source giữ system/user prompt và cách train/chấm của output này.

Kết quả đã kiểm tra từ lịch sử **Versions của cùng notebook**:

| Dataset | Output version | Test Micro-F1 | Best epoch | Early stop |
|---|---:|---:|---:|---:|
| SROIE | 1 | 0.7845543125 | 4 | 9 |
| CORD-v2 | 3 | 0.8192112456 | 5 | 10 |
| WildReceipt | 5 | 0.7446325351 | 4 | 9 |

Source dataset/trainer và scorer của cả ba version giống nhau từng byte. Cả 9 split và 3 bộ nhãn đều khớp benchmark đã chốt. SROIE chạy 1–9; lượt cuối CORD-v2 resume sau epoch 8, WildReceipt sau epoch 6. Output của version mới nhất chỉ chứa WildReceipt; không dùng danh sách latest output để kết luận hai dataset còn lại chưa có kết quả. Report kiểm chứng: `reports/verification-qwen-20261124.json`.

`train.py` chỉ sửa đường dẫn prepared data và thêm bảo vệ không ghi đè checkpoint; `dataset.py` chỉ chuẩn hóa whitespace. SHA nguồn/final trong `reports/benchmarks/provenance.json`. `published-config/` và `requirements-published.txt` lưu cấu hình/khai báo dependency đã xuất. Lượt Kaggle không khóa toàn bộ phiên bản dependency, nên chưa có môi trường GPU tái lập được kiểm chứng trong lần cập nhật này.

Notebook `notebooks/qwen_vl_model.ipynb` mặc định tắt mọi cờ chạy, chọn WildReceipt. Để chủ động train mới cần môi trường CUDA và các dependency trong `requirements-qwen.txt`, pretrained weights và split đã kiểm chứng. Không cài các package GPU này để chỉ mở demo CPU. Không có training được gửi bởi việc cập nhật source.

**Ghi chú cách đánh giá:** prompt đưa vào `generate` không chứa câu trả lời assistant/nhãn chuẩn. Tuy nhiên số token sinh tối đa phụ thuộc batch nhãn chuẩn: `min(1024, max(answer_token_count) + 32)`. Vì vậy đây là điểm công bố theo protocol hiện có, không mô tả là evaluation hoàn toàn không dùng thông tin nhãn chuẩn. Chưa chấm lại với token budget độc lập nhãn.

Kaggle có liệt kê `best_weights.pt`, `last_checkpoint.pt` và `checkpoints.zip`. Chúng chưa được tải/nạp trong lần cập nhật này; `best_weights.pt` là adapter LoRA, cần đúng base model và LoRA config trong trainer, không nạp như checkpoint đầy đủ của năm model trong demo. Demo tương tác hiện vẫn hỗ trợ năm model đã có 30 ví dụ chạy thật; Qwen mới được bổ sung source và kết quả công bố.
