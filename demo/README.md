# Demo hóa đơn

Chạy `setup.bat` một lần rồi `run.bat` ở thư mục gốc. App tại <http://127.0.0.1:8766>. Mặc định dùng CPU. Có thể đặt `RECONIA_DEVICE=cuda` nếu đã cài PyTorch CUDA và GPU đủ bộ nhớ.

Chọn model/dataset để xem một trường đúng và một trường sai trên hai hóa đơn test. `examples.json` chứa prediction đã kiểm chứng; `samples.json`, `sample_images/` chứa OCR, nhãn chuẩn và ảnh. Đây là mẫu từ output Kaggle đã chốt ngày 21/11/2026, không phải toàn bộ test. Một thẻ ghi “Trường đúng” không có nghĩa mọi trường của hóa đơn đều đúng.

**Chạy** nạp checkpoint và suy luận lại. **Xem heatmap** giải thích một trường trong dự đoán. Ảnh ngoài hỗ trợ JPG/PNG/WebP tối đa 12 MB, 40 megapixel. RoBERTa, BROS, PICK, LayoutLMv3 dùng RapidOCR cho ảnh ngoài; Donut đọc ảnh trực tiếp. Mẫu test dùng OCR chú thích. Nhãn chuẩn không đi vào hàm prediction.

## Tài nguyên

Code import từ `models/`, nhãn từ `demo/metadata/`, config/tokenizer từ `demo/model_assets/`. Không nạp code từ `.cache` và không tải pretrained weights khi suy luận. Checkpoint tại `checkpoints/<dataset>/<model>/best_weights.pt`, hoặc dưới `RECONIA_CHECKPOINT_DIR`; phải khớp SHA-256/kích thước trong `artifacts.json` trước khi nạp.

Máy mới có thể xem ví dụ/ảnh/điểm đã lưu sau cài môi trường; suy luận thật cần checkpoint. Xem README gốc để tải. `external-assets.json` ghim file lớn bằng hash; tool tải chỉ đọc output đã hoàn tất, không tạo job.

## Heatmap

Dùng **che vùng (Occlusion)** cho **một trường**, hỗ trợ PICK, LayoutLMv3 và Donut. Chọn trường trong mục Heatmap hoặc để Tự chọn. Nhiều giá trị cùng trường được giải thích chung; trường bị bỏ sót không có đáp án để giải thích.

Mỗi ô được thay bằng ảnh làm mờ rồi chạy lại để đo mức giảm điểm của **cùng đáp án dự đoán ban đầu**. LayoutLMv3/PICK có chế độ **Ảnh + chữ OCR** hoặc **Chỉ ảnh**, giữ vị trí/độ dài chuỗi OCR cố định. Donut chấm token giá trị do chính model sinh, giữ prefix. PICK dùng xác suất biên CRF; LayoutLMv3 dùng xác suất nhãn tại các từ được chọn trong dự đoán gốc.

Lưới 7×7 cho LayoutLMv3/PICK, 5×5 cho Donut được nội suy lên ảnh. Donut được bỏ padding và đảo xoay về tọa độ gốc. Chỉ vẽ giảm điểm dương lớn hơn nhiễu lặp, không khuếch đại tín hiệu yếu thành đỏ đậm. Vùng đỏ hỗ trợ đáp án đang xem, không chỉ đúng/sai hay ánh nhìn. Đây là độ nhạy theo vùng, phụ thuộc baseline làm mờ, không đo hết tương tác vùng hoặc hình học OCR. Không dùng nhãn chuẩn ép vùng nóng. Grad-CAM cũ không còn trong luồng demo.

Heatmap cần nhiều forward pass, lần đầu có thể chậm. Cache tại `.cache/demo-occlusion-heatmaps/` theo ảnh/OCR/checkpoint/trường/scope/phiên bản thuật toán; xem lại báo “Heatmap đã tính”. Xóa cache vẫn tạo lại được. Nút Chạy vẫn dự đoán thật; trọng số không được cập nhật.

`python -m tools.verify_final` kiểm tra source, 15 bộ metrics và 30 ví dụ. `python -m tools.check_demo_predictions` chạy lại 30 prediction. Báo cáo mới sinh tại `output/verification/`; bằng chứng kiểm toán cũ trong cache local không phải phụ thuộc của app.

Source Qwen và ba kết quả SROIE/CORD-v2/WildReceipt từ Versions được bổ sung ngày 24/11/2026 trong `models/qwen_vl/`, `reports/benchmarks/`. Qwen chưa nằm trong bộ lọc demo vì checkpoint/base model và inference chưa được tích hợp, kiểm chứng ở local.
