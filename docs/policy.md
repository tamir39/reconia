# Quy Chuẩn Mã Nguồn

## 1. Phân bổ module
- Thư mục `prepare/` xử lý dữ liệu thô và xuất tệp JSONL vào `.cache/<dataset>/`.
- Thư mục `evaluate/` tính toán chỉ số đánh giá, không chứa thao tác đọc ghi tệp.
- Thư mục `models/<model>/` chứa mã nguồn huấn luyện và nạp dữ liệu độc lập của từng mô hình.
- Thư mục `tools/` chứa các tiện ích thực thi chung.

## 2. Định dạng và định danh
- Tuân thủ PEP 8 về thụt lề và khoảng cách dòng.
- Sắp xếp import theo 3 khối: thư viện chuẩn, thư viện bên thứ ba, module nội bộ.
- Dùng tiếng Anh cho toàn bộ định danh trong mã nguồn, log console và chú thích.
- Đặt tên theo quy ước `snake_case` cho hàm và biến, `PascalCase` cho lớp, `UPPER_SNAKE_CASE` cho hằng số.
- Chú thích ngắn gọn, tập trung vào lý do kỹ thuật hoặc thuật toán, không mô tả lại cú pháp hiển nhiên.

## 3. Kiểu dữ liệu (Python 3.10+)
- Dùng kiểu dựng sẵn chữ thường: `list`, `dict`, `tuple`, `set`.
- Dùng toán tử `|` cho kiểu kết hợp hoặc giá trị tùy chọn (`str | None`, `int | float`).
- Không import từ thư viện `typing`.

```python
# Đúng
def process(items: list[str], cfg: dict[str, float] | None) -> tuple[int, int]:
    ...
```

## 4. Cấu trúc hàm và an toàn dữ liệu
- Viết hàm phẳng, ưu tiên thoát sớm bằng guard clauses.
- Hàm xử lý dữ liệu không lưu trạng thái. Chỉ dùng lớp khi quản lý mô hình hoặc dữ liệu nạp theo batch.
- Kiểm tra điều kiện biên để phòng tránh lỗi chia cho 0, truy xuất khóa từ điển bằng `.get()` và đối chiếu độ dài trước khi ghép cặp.
- Cố định seed ngẫu nhiên (`seed=42`) tại các bước phân chia dữ liệu và khởi tạo huấn luyện.
- Không phòng thủ mù quáng hoặc kiểm tra các giá trị không tồn tại trong dữ liệu thực tế. Không dùng `try...except pass` để nuốt lỗi; ném ngoại lệ rõ ràng khi thiếu tệp hoặc sai cấu trúc dữ liệu.

## 5. Chuẩn gán nhãn dữ liệu
- Sử dụng chuẩn gán nhãn BIO (IOB2: `B-<field>`, `I-<field>`, `O`) cho toàn bộ tác vụ phân loại token và gán nhãn chuỗi.
- Ký hiệu `O` đại diện duy nhất cho các từ hoặc hộp ngoài lề, không sử dụng chuỗi thay thế tự chế.

## 6. Quy ước commit và tóm tắt thay đổi
- Áp dụng cấu trúc `<loại>: <mô tả>` với các tiền tố chính: `feat`, `fix`, `refactor`, `docs`, `perf`.
- Mô tả trực tiếp hành động kỹ thuật cụ thể, không dùng câu đánh giá chung chung hoặc văn mẫu sáo rỗng.
- Viết ở thể mệnh lệnh, không dùng đại từ nhân xưng ngôi thứ nhất, độ dài tiêu đề không quá 72 ký tự.
