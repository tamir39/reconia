# Huấn luyện GNN trên Kaggle

## RoBERTa trên cùng DocILE split

Notebook mới: `notebooks/kaggle_roberta_docile.ipynb`, mặc định **RUN_MODE = "train"**. Có một phiên smoke riêng để kiểm tra trước khi bắt đầu thực nghiệm đầy đủ; đọc dòng MODE và số tài liệu trong log để phân biệt.

RoBERTa-base + nhúng bố cục 2D dùng checkpoint chính thức `FacebookAI/roberta-base`, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Cấu hình ban đầu: 20 epochs, batch 4 × accumulation 4, backbone LR 5e-5, head LR 2e-4, seed 42, patience 5. Mọi từ OCR được xử lý qua các cửa sổ 512 token, mỗi từ nhận một nhãn ở subword đầu tiên; dự đoán được ghép theo chỉ số từ và trang. Không còn cắt bỏ phần cuối tài liệu. Chính sách cửa sổ được lưu cùng checkpoint; số bước mỗi epoch phụ thuộc số cửa sổ, không chỉ số tài liệu.

Best/latest checkpoint lưu kèm tokenizer để có thể nạp lại ngoài phiên Kaggle ban đầu. Bundle dùng chung với GNN; mã được cập nhật bằng dataset version mới. Trọng số backbone được tải trực tiếp từ Hugging Face vào cache tạm. Đầu ra RoBERTa nằm trong thư mục riêng, không ghi đè kết quả GNN.

GNN đã có kết quả được kiểm chứng: entity F1 66,82%, field-value F1 56,87%; xem [bảng kết quả đã kiểm chứng](../reports/verified_comparison.md). Không dùng các báo cáo RoBERTa lịch sử để kết luận so sánh với lần GNN này.

## GNN

Notebook: `notebooks/kaggle_gnn_docile.ipynb`. Lần chạy đầu mặc định `RUN_MODE = "smoke"` để kiểm tra GPU, dữ liệu, train 16 mẫu và nạp lại checkpoint. Sau khi thành công, đổi thành `"train"` và chọn **Save Version → Save & Run All** để chạy toàn bộ. Không dùng điểm smoke test trong báo cáo.

## Thiết lập

- Notebook và dataset đều để private. Chọn GPU và bật Internet để cài các thư viện Python cần thiết. Mã hiện dùng một GPU, kể cả khi máy có hai GPU.
- Attach dataset **Reconia DocILE GNN Training Bundle**. Notebook tự tìm `reconia-code.bin` và `docile-export.bin`, giải nén, kiểm tra checksum và so khớp ID với split trong mã nguồn. Hai file là ZIP dùng đuôi `.bin` để Kaggle không tự giải nén và thay đổi tên/hash của các shard gzip bên trong.
- Bundle chứa OCR/nhãn, không chứa ảnh. Chỉ dùng cho GNN hiện tại; LayoutLMv3/VLM cần dữ liệu ảnh riêng.
- Dùng PyTorch/CUDA có sẵn của Kaggle; không chạy toàn bộ `requirements.txt` của ứng dụng web trong notebook.

## Chạy đầy đủ

Giữ mặc định: 20 epochs, seed 42, learning rate 0.001, hidden dimension 128, 2 layers, gradient accumulation 16, early stopping patience 5. Đây là cấu hình khởi đầu, chưa phải tham số tối ưu đã được chứng minh. Dùng val để chọn checkpoint; test nội bộ 500 tài liệu chỉ dùng sau khi chọn mô hình.

Mỗi lần chạy ghi `run-config.json`, phiên bản thư viện, nguồn mã, log, best/latest checkpoint và các báo cáo vào `/kaggle/working/reconia-runs/<RUN_ID>`. Graph cache nằm trong `/kaggle/temp`, không nằm trong ZIP kết quả. Đổi RUN_ID khi bắt đầu thí nghiệm mới; notebook chặn ghi đè checkpoint đã có.

Full training tự đánh giá test và xuất dự đoán cùng `.metrics.json`. Notebook kiểm tra `complete=true` và đủ 500 tài liệu rồi tạo ZIP kết quả trong `/kaggle/working`. Tải ZIP sau khi Save & Run All hoàn tất. Không chỉ dựa vào phiên interactive để giữ trọng số.

## Tiếp tục phiên bị ngắt

Attach output đã lưu của lần chạy trước. Đặt `RESUME_FROM` bằng đường dẫn thư mục `checkpoints` chứa **cả** `gnn_docile_latest.pt` và `gnn_docile_best.pt`; đặt RUN_ID mới, giữ nguyên tham số và tổng EPOCHS. Notebook sao chép hai file vào thư mục mới trước khi resume, tránh mất checkpoint tốt nhất nếu các epoch tiếp theo không cải thiện. Chỉ checkpoint của các epoch đã hoàn tất được khôi phục; không bảo đảm tiếp tục bit-for-bit như phiên không bị ngắt. Không dùng checkpoint GNN trước bản sửa stable hash.

Khi phiên kết thúc đột ngột, checkpoint chỉ có thể khôi phục nếu file thực sự đã được Kaggle lưu. Luôn kiểm tra Outputs. Xem [tài liệu notebook của Kaggle](https://www.kaggle.com/docs/notebooks).

## Tạo lại bundle từ máy cá nhân

```powershell
python scripts/prepare_kaggle.py --export .cache/week5-export/docile --output .cache/kaggle/upload --owner YOUR_KAGGLE_USERNAME
Push-Location .cache/kaggle/upload
python -m kaggle datasets create -p .
Pop-Location
```

Không thêm `--public`. Script chỉ đóng gói mã và dữ liệu đã chọn, không lấy `.env`, API key hoặc trọng số. Khi dataset đã tồn tại, tạo version mới thay vì tạo bản trùng. Mã nguồn kèm hash trong bundle là nguồn dùng để train; thay đổi GitHub không tự cập nhật notebook đang chạy.
