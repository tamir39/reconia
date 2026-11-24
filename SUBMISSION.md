# Bản chuẩn bị nộp

Nhánh `final-submit`: source **6 model**, demo tương tác 5 model, 15 bộ benchmark Micro-F1 và 30 trường minh họa, cộng đủ ba kết quả công bố của Qwen (cập nhật 24/11/2026). Nguồn và thay đổi chạy local tại `reports/benchmarks/provenance.json`.

Chạy `setup.bat`, lấy checkpoint nếu máy chưa có, rồi `run.bat` theo README. Tạo ZIP bằng `python -m tools.export_submission`. Gửi kèm link nhánh và README hướng dẫn lấy trọng số. ZIP không chứa checkpoint, raw data, `.env`, `.venv` hay cache.

Checkpoint/raw data trên máy được giữ nguyên. Cleanup không tạo lượt train. Source Qwen cũ trong `archive/` giữ để tham khảo. Qwen mới tại `models/qwen_vl/`; kết quả SROIE từ version 1, CORD-v2 từ version 3, WildReceipt từ version 5 của cùng link Thịnh; chưa tích hợp inference Qwen vào demo. Ghi chú protocol trong README của model.

Bản thảo ở `reports/archive/` cần duyệt nội dung/định dạng riêng trước khi nộp báo cáo học thuật; cleanup này không xác nhận bản thảo cũ là báo cáo final.
