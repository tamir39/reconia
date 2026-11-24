# Báo cáo và kết quả

- `benchmarks/`: 15 bộ kết quả của 5 model × 3 dataset và một kết quả WildReceipt công bố của Qwen: test metrics, history, train log và nguồn/hash trong `provenance.json`. Scorer hiện tại là Micro-F1.
- `archive/`: ghi chú, kiểm toán và bản thảo lịch sử. Có tài liệu còn dùng sáu model/Macro-F1; không lấy thay bảng final và không coi là báo cáo học thuật final đã duyệt.

`python -m tools.compare` tạo bảng/biểu đồ từ benchmark đã chốt vào `output/comparison/`. `python -m tools.verify_final` kiểm tra benchmark và dữ liệu minh họa không lệch demo.

`benchmarks/additional-models.json` ghi rõ Qwen chỉ có WildReceipt từ link mới, chưa chạy lại checkpoint, và generation budget phụ thuộc độ dài nhãn chuẩn. Không suy ra hai điểm còn thiếu.
