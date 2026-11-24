# Báo cáo và kết quả

- `benchmarks/`: 15 bộ kết quả của 5 model × 3 dataset và ba kết quả công bố của Qwen: test metrics, history, train log và nguồn/hash trong `provenance.json`. Scorer hiện tại là Micro-F1.
- `archive/`: ghi chú, kiểm toán và bản thảo lịch sử. Có tài liệu còn dùng sáu model/Macro-F1; không lấy thay bảng final và không coi là báo cáo học thuật final đã duyệt.

`python -m tools.compare` tạo bảng/biểu đồ từ benchmark đã chốt vào `output/comparison/`. `python -m tools.verify_final` kiểm tra benchmark và dữ liệu minh họa không lệch demo.

`benchmarks/additional-models.json` ghi rõ Qwen lấy SROIE từ output version 1, CORD-v2 từ version 3, WildReceipt từ version 5; source/scorer của ba version giống nhau và 9 split khớp. Chưa chạy lại checkpoint; generation budget phụ thuộc độ dài nhãn chuẩn.

Ghi chú protocol chung: Donut cũng giới hạn độ dài sinh theo số token nhãn chuẩn khi validation/test (`models/donut/train.py`), tương tự Qwen. Các điểm hiện tại giữ theo protocol đã xuất; demo ảnh ngoài dùng budget cố định.
