# Bàn giao bản nộp source Reconia

Nhánh: [`final-submit`](https://github.com/tamir39/reconia/tree/final-submit). File nộp source: `dist/reconia-final-source.zip`, tạo từ Git checkout bằng `python -m tools.export_submission`.

## Phạm vi đã có trong repo

- Source của 6 model và 6 notebook, mặc định tắt train/download/export.
- 18 bộ test metrics/history/log trên 3 dataset, kèm provenance. Qwen lấy SROIE từ version 1, CORD-v2 từ version 3, WildReceipt từ version 5 của cùng notebook Thịnh.
- Demo tương tác 5 model, 30 trường minh họa đúng/sai; inference ảnh test và ảnh ngoài bằng checkpoint.
- Heatmap Occlusion cho một trường của PICK, LayoutLMv3, Donut. Qwen chưa được tích hợp inference/heatmap vào demo local.
- Hướng dẫn cài/chạy, script kiểm chứng, tests và manifest hash của source.

Đọc ghi chú protocol trong README: Donut/Qwen dùng độ dài đáp án thật để giới hạn token khi đánh giá. Đây là điểm từ các lượt đã xuất, chưa được chấm lại với token budget độc lập nhãn.

## Tài nguyên bàn giao riêng

**Checkpoint:** ZIP source không kèm trọng số. Để người nhận chạy inference thật, cung cấp quyền đọc output Kaggle hoặc bộ 15 checkpoint đúng hash theo `demo/external-assets.json`. Người nhận dùng tài khoản/key riêng; không gửi `.env` hay API key trong source. Qwen cần base model, adapter và môi trường CUDA riêng, chưa có bước nạp local được kiểm chứng.

**Raw dataset:** không kèm ZIP source. Không cần raw data để xem ví dụ hoặc upload ảnh demo; cần ảnh/data gốc nếu chủ động train lại. JSONL tải bằng `tools.fetch_assets --prepared` không thay thế ảnh raw.

**Báo cáo học thuật:** các file trong `reports/archive/` là bản thảo/lịch sử. Cần duyệt nội dung, bảng số liệu, phương pháp heatmap và định dạng báo cáo final riêng; repo source không tự thay thế báo cáo final.

## Kiểm tra trước khi gửi

Chạy `setup.bat`, tests và `python -m tools.verify_final`. Khi có đủ 15 checkpoint, kiểm tra thêm `--weights` và `tools.check_demo_predictions`. Hướng dẫn đầy đủ trong README.

Nộp link nhánh hoặc ZIP do tool xuất, kèm cách lấy checkpoint và báo cáo final đã duyệt nếu yêu cầu môn học. Không nén nguyên workspace vì máy làm việc còn `.env`, cache, runtime, dữ liệu và trọng số lớn. `SOURCE_MANIFEST.json` trong ZIP ghi commit/hash để xác định đúng bản.
