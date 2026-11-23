# Bản chuẩn bị nộp

Nhánh `final-submit`: source/demo **5 model**, 15 bộ benchmark Micro-F1, 30 trường minh họa đúng/sai. Nguồn và thay đổi chạy local tại `reports/benchmarks/provenance.json`.

Chạy `setup.bat`, lấy checkpoint nếu máy chưa có, rồi `run.bat` theo README. Tạo ZIP bằng `python -m tools.export_submission`. Gửi kèm link nhánh và README hướng dẫn lấy trọng số. ZIP không chứa checkpoint, raw data, `.env`, `.venv` hay cache.

Checkpoint/raw data trên máy được giữ nguyên. Cleanup không tạo lượt train. Qwen/source sáu model cũ được lưu trong `archive/`, không thuộc benchmark final năm model.

Bản thảo ở `reports/archive/` cần duyệt nội dung/định dạng riêng trước khi nộp báo cáo học thuật; cleanup này không xác nhận bản thảo cũ là báo cáo final.
