# DỮ LIỆU VÀ CÁCH CÁC MÔ HÌNH SỬ DỤNG

**Đề tài:** Đánh giá và so sánh các phương pháp học sâu cho bài toán trích xuất thông tin
từ hóa đơn (ĐA161)

Tài liệu mô tả bộ dữ liệu đang sử dụng, các định dạng trung gian mà nhóm xây dựng, và cách
từng nhóm mô hình tiêu thụ dữ liệu đó.

Số liệu khảo sát dưới đây được lưu từ giai đoạn chuẩn bị dữ liệu với 5.180 train / 500 val. Từ tuần 5, chia tập nội bộ là 4.680 train / 500 val / 500 test; không dùng bản xuất cũ làm tập train khi đánh giá trên 500 tài liệu test mới. Cần xuất lại theo `docile_splits.json`.

---

## 1. Bộ dữ liệu

### 1.1. DocILE — bộ dữ liệu chính

Phần `annotated-trainval` của bộ dữ liệu DocILE (Šimša và cộng sự, ICDAR 2023).

| Chỉ tiêu | Tập huấn luyện | Tập kiểm định |
|---|---|---|
| Số tài liệu | 5.180 | 500 |
| Số từ | 1.515.895 | 149.954 |
| Số trường được chú giải | 212.477 | 16.505 |

Mỗi tài liệu gồm ba thành phần:

| Thành phần | Nội dung |
|---|---|
| `annotations/<id>.json` | Vị trí và loại của từng trường thông tin |
| `ocr/<id>.json` | Kết quả OCR sẵn có, kèm tọa độ từng từ |
| `pdfs/<id>.pdf` | Tệp PDF gốc |

Kết quả OCR do nhóm tác giả bộ dữ liệu tạo sẵn bằng thư viện DocTR, kèm hộp bao đã được
căn lại theo vùng chữ. Việc dùng chung kết quả OCR này cho mọi mô hình loại bỏ một nguồn
sai khác khi so sánh: chênh lệch giữa các mô hình không bị lẫn với chênh lệch do chất
lượng nhận dạng đầu vào.

### 1.2. Bộ dữ liệu bổ sung

Đã có adapter và chuẩn bị cục bộ ảnh/nhãn của **SROIE** (626 receipt, split nội bộ
500/63/63) và **CORD-v2** (800/100/100 theo nguồn). Xem
[nguồn, mapping SROIE, kiểm tra và bundle Kaggle](RECEIPT_PREPARATION.md).
Đây là dữ liệu đầu vào, chưa phải kết quả huấn luyện mới. Mapping numeric ID của
SROIE cũ chưa được xác nhận; so sánh VLM cần tích hợp chấm key gốc thay vì đoạn OCR.

---

## 2. Đặc điểm dữ liệu ảnh hưởng tới thiết kế

Ba đặc điểm dưới đây được đo trước khi xây dựng mô hình, và mỗi đặc điểm đều dẫn tới một
quyết định thiết kế cụ thể.

### 2.1. Trần lý thuyết của phương pháp gán nhãn theo từ: 94,42%

Một trường chỉ được coi là biểu diễn được nếu có ít nhất một từ do OCR trả về nằm trong
vùng của trường đó.

| Tập | Số trường | Biểu diễn được | Tỷ lệ |
|---|---|---|---|
| Huấn luyện | 212.477 | 200.631 | **94,42%** |
| Kiểm định | 16.505 | 15.604 | **94,54%** |

Có **11.846 trường** trong tập huấn luyện mà không từ nào biểu diễn được. Nguyên nhân chủ
yếu là ký hiệu tiền tệ bị OCR gộp vào con số, và một số trường mà hộp chú giải không trùng
với vùng chữ.

**Hệ quả:** mọi mô hình gán nhãn ở mức từ — gồm mô hình đồ thị, RoBERTa và LayoutLMv3 —
đều bị chặn trên ở mức 94,4%. Các phương pháp sinh văn bản trực tiếp không chịu giới hạn
này. Khi so sánh hai nhóm phương pháp, con số này phải được nêu, nếu không việc so sánh
sẽ không công bằng.

Hai tập lệch nhau 0,12%, cho thấy ước lượng ổn định.

### 2.2. 25,9% số trường trải trên nhiều từ

| Số từ tạo nên một trường | Tỷ lệ |
|---|---|
| 1 từ | 73,7% |
| 2 từ | 7,1% |
| 3 từ | 6,0% |
| 4 từ | 3,0% |
| 5 từ | 2,0% |
| Từ 6 từ trở lên | 8,2% |

**Hệ quả:** không thể gán nhãn từng từ độc lập. Hai trường cùng loại nằm cạnh nhau, ví dụ
hai mã hàng liên tiếp, sẽ không tách được khi ghép các từ cùng nhãn lại. Nhóm dùng lược đồ
**BIO**: từ đầu tiên của mỗi trường gán tiền tố `B-`, các từ tiếp theo gán `I-`.

### 2.3. Phân bố nhãn mất cân bằng nghiêm trọng

Trong 65.651 trường KILE của tập huấn luyện:

| Loại trường phổ biến nhất | Số lượng | Tỷ lệ |
|---|---|---|
| `vendor_name` | 6.714 | 10,23% |
| `vendor_address` | 6.088 | 9,27% |
| `date_issue` | 5.702 | 8,69% |
| `document_id` | 5.688 | 8,66% |
| `customer_billing_name` | 5.629 | 8,57% |

| Loại trường hiếm nhất | Số lượng | Tỷ lệ |
|---|---|---|
| `bic` | 31 | 0,05% |
| `customer_registration_id` | 8 | 0,01% |
| `iban` | 5 | 0,01% |

Mười loại trường phổ biến nhất chiếm **79,9%** tổng số trường KILE. Loại hiếm nhất chỉ
xuất hiện **5 lần trong 5.180 tài liệu**.

**Hệ quả:** phải áp dụng trọng số theo lớp khi huấn luyện, và phải báo cáo **macro-F1**
bên cạnh micro-F1. Nếu chỉ báo cáo độ chính xác tổng thể, một mô hình luôn dự đoán nhãn
`O` vẫn đạt con số rất cao mà không trích xuất được gì.

### 2.4. Quy mô tài liệu

| Chỉ tiêu | Giá trị |
|---|---|
| Số từ mỗi trang | trung bình 293, trung vị 199, phân vị 95 là 801, lớn nhất 4.106 |
| Số trang | 1 trang 74,6% · 2 trang 20,3% · 3 trang 5,1% |
| Số dòng hàng mỗi tài liệu | trung bình 7,0, lớn nhất 110 |
| Số loại trường | KILE 36 loại · LIR 19 loại · tổng 55 |

Trung vị 199 từ cho thấy phần lớn tài liệu khá nhỏ; phân vị 95 ở mức 801 cho thấy có một
đuôi dài. Điều này ảnh hưởng tới cách gộp lô khi huấn luyện: gộp theo số tài liệu sẽ tạo
ra các lô có kích thước rất chênh lệch.

---

## 3. Các định dạng dữ liệu trong hệ thống

```
Bộ dữ liệu gốc
  DocILE 1,7 GB, 17.048 tệp
        │
        ▼  DatasetAdapter
Định dạng trung gian          app/data/schema.py
  Document: words + fields
        │
        ├──────────────────────────────┬───────────────────────┐
        ▼                              ▼                       ▼
  Bản xuất chia mảnh              Đồ thị                  Chuỗi token
  31 MB, 40 tệp                   app/graph/              (tuần 4 trở đi)
  app/data/export.py              293 nút, 1.142 cạnh
        │                              │                       │
        ▼                              ▼                       ▼
   Tải lên Kaggle              Mô hình đồ thị        RoBERTa, LayoutLMv3
```

### 3.1. Định dạng trung gian

Định nghĩa trong `app/data/schema.py`. Mọi bộ dữ liệu được chuyển về định dạng này, và
mọi mô hình đọc từ định dạng này.

| Kiểu | Nội dung |
|---|---|
| `BBox` | Hộp bao, tọa độ chuẩn hóa về đoạn [0, 1] |
| `Word` | Một từ do OCR trả về: nội dung, hộp bao, số trang, độ tin cậy |
| `Field` | Một trường được chú giải: loại, hộp bao, số trang, nội dung, mã dòng hàng |
| `Document` | Một tài liệu: danh sách từ và danh sách trường |

Tọa độ luôn được chuẩn hóa về đoạn [0, 1] theo kích thước trang, nên tài liệu có kích
thước và độ phân giải khác nhau vẫn so sánh trực tiếp được.

**Đây là cơ chế bảo đảm điều kiện thực nghiệm thống nhất mà mô tả đề tài yêu cầu.** Mọi mô
hình đọc cùng một nguồn dữ liệu, nên chắc chắn nhận cùng từ, cùng tọa độ và cùng nhãn.

### 3.2. Bản xuất chia mảnh

Sinh bằng `app/data/export.py`, đặt tại `thư mục xuất được truyền qua `--output``.

| Chỉ tiêu | Giá trị |
|---|---|
| Dung lượng | 31 MB (từ 1,7 GB dữ liệu gốc) |
| Số tệp | 40 mảnh nén, mỗi tập 20 mảnh |
| Định dạng | JSON Lines nén gzip, mỗi dòng một tài liệu |
| Thời gian đọc lại | 500 tài liệu trong 0,6 giây |

Hai đặc điểm có chủ đích:

- **Chia mảnh theo vòng tròn.** Tài liệu thứ `i` được ghi vào mảnh `i mod 20`, không ghi
  theo khối liên tiếp. Nhờ vậy mỗi mảnh là một mẫu đại diện, và một bản tải về dở dang vẫn
  dùng được.
- **Mã băm SHA-256 cho từng mảnh**, lưu trong `manifest.json`. Sau khi tải lên Kaggle cần
  chạy `verify()`; một bản truyền thiếu sẽ hiện ra dưới dạng lỗi thay vì dưới dạng sụt
  giảm độ chính xác không rõ nguyên nhân.

Đã kiểm chứng bản xuất không mất mát: so với dữ liệu gốc, nội dung từ trùng khớp hoàn
toàn, loại trường trùng khớp hoàn toàn, mã dòng hàng được giữ nguyên, sai lệch tọa độ lớn
nhất là 5×10⁻⁶ do làm tròn về 5 chữ số thập phân.

### 3.3. Biểu diễn đồ thị

Sinh bằng `app/graph/builder.py`, lưu đệm tại `thư mục được truyền qua `--cache-dir``.

| Thành phần | Định nghĩa |
|---|---|
| Nút | Một từ do OCR trả về |
| Đặc trưng nút | 17 giá trị: tọa độ, kích thước, tâm, tỷ lệ, diện tích, độ dài, đặc điểm ký tự, độ tin cậy |
| Cạnh | Ba chiến lược, xem mục 3.4 |
| Nhãn nút | Lược đồ BIO, 111 nhãn sinh từ 55 loại trường |

| Chỉ tiêu | Tập huấn luyện | Tập kiểm định |
|---|---|---|
| Nút mỗi tài liệu | 293 | 300 |
| Cạnh mỗi tài liệu | 1.142 | 1.184 |
| Cạnh trên mỗi nút | 3,9 | 3,9 |

Thời gian dựng 127 ms mỗi tài liệu. Sau khi lưu đệm, nạp lại chỉ mất 1,3 ms mỗi tài liệu,
tức khoảng 7 giây cho một vòng lặp huấn luyện trên toàn bộ tập huấn luyện.

Thư mục lưu đệm được đặt tên theo cấu hình tạo cạnh, ví dụ `li1-knn6-line1-vert1`. Nhờ vậy
một thí nghiệm chạy với cấu hình cạnh khác không thể vô tình dùng lại đồ thị cũ, điều sẽ
làm sai lệch kết quả so sánh.

### 3.4. Ba chiến lược tạo cạnh

| Chiến lược | Cách nối | Vai trò |
|---|---|---|
| `knn` | Nối mỗi từ với k từ gần nhất theo khoảng cách tâm | Bảo đảm không nút nào bị cô lập |
| `same_line` | Nối các từ liền kề trên cùng một dòng | Mang cấu trúc theo dòng |
| `vertical` | Nối mỗi từ với từ nằm ngay bên dưới | Mang cấu trúc theo cột |

Khi nhiều chiến lược cùng sinh ra một cạnh, quan hệ cấu trúc được giữ lại thay cho quan hệ
khoảng cách: `same_line` và `vertical` mô tả bố cục tài liệu, còn `knn` chỉ mô tả sự gần nhau.

Trên toàn bộ tập huấn luyện, khoảng 39% số cạnh do `knn` sinh ra bị hai chiến lược cấu trúc
ghi đè, cho thấy đồ thị phản ánh bố cục thật chứ không chỉ phản ánh khoảng cách.

---

## 4. Cách từng nhóm mô hình sử dụng dữ liệu

### 4.1. Mô hình đồ thị (nhóm hiện thực)

```
Document ──► build_graph() ──► đặc trưng nút, danh sách cạnh, nhãn BIO ──► GCN
```

- **Đầu vào:** ma trận đặc trưng `(số nút, 17)`, danh sách cạnh `(2, số cạnh)`, loại cạnh.
- **Đầu ra:** một nhãn BIO cho mỗi nút, chọn trong 111 nhãn.
- **Hậu xử lý:** ghép các từ liền kề cùng nhãn thành trường hoàn chỉnh, bằng
  `decode_fields()` trong `app/graph/labels.py`.
- **Nơi chạy:** máy cá nhân. Đồ thị trung bình 293 nút nên GPU 4 GB là đủ.

### 4.2. Mô hình theo chuỗi — RoBERTa

```
Document ──► sắp từ theo thứ tự đọc ──► token ──► RoBERTa ──► nhãn BIO
```

- **Đầu vào:** chuỗi từ theo thứ tự đọc, không dùng thông tin tọa độ.
- Thứ tự đọc lấy từ `reading_order()` trong `app/graph/labels.py`, nên hoàn toàn trùng với
  thứ tự mà mô hình đồ thị dùng để gán nhãn.
- **Nơi chạy:** Kaggle.
- **Số liệu công bố trên DocILE:** val-AP 0,552 · val-F1 0,688.

### 4.3. Mô hình đa phương thức — LayoutLMv3

```
Document + ảnh trang ──► token + tọa độ + mảng ảnh ──► LayoutLMv3 ──► nhãn BIO
```

- **Đầu vào:** token văn bản, tọa độ hộp bao đã chuẩn hóa về thang 0–1000, và ảnh trang.
- Đây là mô hình duy nhất cần tới tệp PDF gốc; các mô hình còn lại chỉ cần bản xuất 31 MB.
- **Nơi chạy:** Kaggle.
- **Số liệu công bố trên DocILE:** val-AP 0,513 · val-F1 0,657.

### 4.4. Điểm cần chú ý trong kết quả công bố

Trên bài toán KILE, **RoBERTa chỉ dùng văn bản lại cho kết quả cao hơn LayoutLMv3 dùng cả
văn bản, bố cục và hình ảnh** (0,552 so với 0,513). Đây là kết quả ngược với kỳ vọng thông
thường và là một trong những nội dung đáng phân tích nhất của đề tài.

Nhóm không giả định trước thứ hạng của các mô hình.

---

## 5. Vị trí lưu trữ

| Đường dẫn | Nội dung | Dung lượng |
|---|---|---|
| `$DOCILE_DIR` | Bộ dữ liệu gốc; đường dẫn được cấu hình riêng trên từng máy | 1,7 GB |
| `thư mục xuất được truyền qua `--output`/` | Bản xuất chia mảnh, dùng để tải lên Kaggle | 31 MB |
| `thư mục được truyền qua `--cache-dir`/` | Đồ thị đã dựng và từ điển nhãn | 95 MB |

Toàn bộ dữ liệu đặt **ngoài kho mã nguồn**. Đường dẫn bộ dữ liệu gốc cấu hình qua biến môi
trường `DOCILE_DIR`.

## 6. Cách tái lập

```bash
# Dựng từ điển nhãn và lưu đệm đồ thị
python -m app.graph.prepare --dataset docile --splits train val

# Xuất bản chia mảnh để tải lên Kaggle
python -c "from app.data.docile import DocileAdapter; from app.data.export import export_dataset; \
export_dataset(DocileAdapter(), 'thư mục xuất được truyền qua `--output`')"

# Kiểm tra tính toàn vẹn sau khi truyền tệp
python -c "from app.data.export import verify; print(verify('thư mục xuất được truyền qua `--output`') or 'OK')"
```
