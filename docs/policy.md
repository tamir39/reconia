# Quy chuẩn dự án

## 1. Phân bổ module
- Thư mục `prepare/`: Chuyển đổi dữ liệu thô sang định dạng JSONL trong `.cache/<dataset>/`.
- Thư mục `evaluate/`: Tính toán chỉ số đánh giá trường thông tin (field-level F1), không thao tác đọc ghi tệp.
- Thư mục `models/<model>/`: Chứa mã nguồn mô hình, nạp dữ liệu và huấn luyện độc lập cho từng kiến trúc.
- Thư mục `tools/`: Các tiện ích dùng chung (cấu hình đường dẫn, trực quan hóa, so sánh kết quả).
- Thư mục `notebooks/`: Sổ tay Jupyter thực nghiệm tương ứng cho từng mô hình.

## 2. Định dạng mã nguồn và định danh
- Tuân thủ PEP 8 về thụt lề và khoảng cách dòng.
- Thứ tự import phân định 3 nhóm: thư viện chuẩn -> thư viện bên thứ ba -> module nội bộ.
- Sử dụng tiếng Anh cho toàn bộ định danh trong mã nguồn, log console và chú thích.
- Quy ước đặt tên: `snake_case` cho hàm và biến, `PascalCase` cho lớp, `UPPER_SNAKE_CASE` cho hằng số.
- Chú thích mã nguồn súc tích, chỉ giải thích lý do kỹ thuật hoặc công thức thuật toán đặc thù.

## 3. Kiểu dữ liệu (Python 3.10+)
- Sử dụng kiểu dựng sẵn chữ thường: `list`, `dict`, `tuple`, `set`.
- Sử dụng toán tử `|` cho kiểu kết hợp hoặc giá trị tùy chọn (`str | None`, `int | float`).
- Không import từ thư viện `typing`.

## 4. Thiết kế hàm và xử lý ngoại lệ
- Viết hàm phẳng, ưu tiên thoát sớm bằng guard clauses.
- Hàm xử lý dữ liệu bảo đảm tính thuần túy (stateless). Chỉ sử dụng lớp cho mô hình hoặc quản lý nạp dữ liệu theo batch.
- Kiểm tra điều kiện biên để tránh chia cho 0 và đối chiếu kích thước dữ liệu trước khi ghép cặp.
- Cố định seed ngẫu nhiên (`seed=42`) khi phân chia dữ liệu và khởi tạo huấn luyện.
- Xử lý ngoại lệ minh bạch: Không bắt ngoại lệ im lặng (`silent fail`); ném ngoại lệ tường minh khi thiếu tệp hoặc sai định dạng dữ liệu.

## 5. Chuẩn gán nhãn dữ liệu
- Sử dụng chuẩn gán nhãn BIO (IOB2: `B-<field>`, `I-<field>`, `O`) cho tác vụ phân loại token và gán nhãn chuỗi.
- Ký hiệu `O` đại diện cho các từ hoặc bounding box không thuộc thực thể cần trích xuất.

## 6. Quy ước commit
- Áp dụng cấu trúc `<loại>: <mô tả>` với các tiền tố: `feat`, `fix`, `refactor`, `docs`, `perf`.
- Mô tả trực tiếp hành động kỹ thuật cụ thể, súc tích và chính xác.
- Viết ở thể mệnh lệnh, không dùng đại từ nhân xưng, độ dài tiêu đề không quá 72 ký tự.