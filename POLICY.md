# POLICY — Reconia (ĐA161)

Nhóm: Phí Vương Tường Tâm (52300155) · Vũ Hưng Thịnh (52300159)
Đề tài: Xây dựng hệ thống đối soát hóa đơn và chứng từ nhập xuất

---

## 1. Nhánh

| Nhánh | Dùng để |
|---|---|
| `main` | Bản nộp. Chỉ merge ở mốc **giữa kỳ** và **cuối kỳ**. Không commit trực tiếp. |
| `dev` | Nhánh làm việc chính. Mọi thứ vào đây. |
| `feature/<tên>/<phần>` | Chỉ tạo khi làm việc lớn hoặc dễ vỡ. Việc nhỏ commit thẳng `dev`. |

`main` cũ hơn `dev` là chuyện bình thường, không phải lỗi.

## 2. Commit message

```
<tên>/<phần>/<việc đã làm>
```

- `<tên>` — `tam` hoặc `thinh`, viết thường không dấu
- `<phần>` — lấy trong bảng mục 3
- `<việc đã làm>` — tiếng Anh, thể mệnh lệnh, không viết hoa đầu, không chấm cuối

```
tam/extraction/add XML e-invoice parser
thinh/matching/add tier-0 explicit reference lookup
tam/db/add direction column for inbound and outbound
thinh/rules/add line arithmetic validation
```

Có tên người trong commit vì **bảng phân công trong báo cáo phải khớp `git shortlog -sn`**,
và GVPB hỏi chéo từng người khi bảo vệ.

> **Làm ngay trên cả hai máy** (không sửa được về sau):
> ```
> git config user.name "Phi Vuong Tuong Tam"
> git config user.email "<email của bạn>"
> ```
> Email nào cũng được — Khoa không quy định. Chỉ cần **mỗi người dùng đúng một
> email cố định trên mọi máy**, vì `git shortlog -sn` gộp theo tên + email.
> Dùng 2 email khác nhau thì một người sẽ hiện thành 2 dòng, bảng phân công trông sai.

## 3. Danh sách `<phần>`

| Phần | Tương ứng |
|---|---|
| `intake` | Module 1 — tiếp nhận, phân loại, tiền xử lý |
| `extraction` | Module 2 — trích xuất, chuẩn hóa |
| `matching` | Module 3 — đối soát, ghép nối |
| `rules` | Module 4 — ràng buộc, phát hiện bất thường |
| `agent` | Module 5 — Agent điều phối, HITL |
| `audit` | Module 6 — nhật ký, bộ nhớ hiệu chỉnh |
| `ui` | Giao diện, màn hình rà soát, báo cáo |
| `db` | Lược đồ CSDL, migration |
| `infra` | Docker, cấu hình |
| `eval` | Dữ liệu mẫu, đo đạc, so sánh mô hình |
| `docs` | Tài liệu, báo cáo |

Thư mục code chia theo đúng danh sách này.

## 4. Push & merge

- Push nhánh của mình **cuối mỗi ngày làm việc**, kể cả chưa xong. Máy hỏng là mất trắng,
  và `git log --since="1 week ago"` chính là nội dung email báo cáo tuần.
- **Không bao giờ** `git push --force` lên `main` hoặc `dev`.
- Merge vào `dev`: **nhắn cho người kia trước**, không cần PR. Nhưng bắt buộc phải nhắn —
  hai người sẽ đụng cùng file.
- Merge vào `main`: **chỉ 2 lần** — nộp giữa kỳ, nộp cuối kỳ. Cả hai cùng đồng ý.

## 5. Quy ước code

- **Tiền luôn dùng `Decimal`, tuyệt đối không dùng `float`.** Cột CSDL dùng `NUMERIC`.
  Sai chỗ này thì mọi kết quả đối soát đều không tin được.
- Mã số thuế là khóa định danh đối tác. **Không bao giờ so khớp theo địa chỉ.**
- Không chạy OCR khi chứng từ đã có XML hoặc đã có lớp văn bản.
- Tên bảng và cột: `snake_case`, tiếng Anh, số ít (`invoice`, `stock_document`, `product_alias`).
- Code và comment tiếng Anh. Nội dung hiển thị cho người dùng tiếng Việt.
- Mỗi quy tắc kiểm tra là **một hàm thuần**, ghi kết quả vào bảng `finding`.

## 6. Một task coi là XONG khi

1. Chạy được trên máy người làm, không lỗi.
2. Có ít nhất 1 test cho phần logic (quy tắc, chuẩn hóa, ghép nối).
3. Đã commit đúng format và push lên remote.
4. Đã nhắn cho người kia biết.
5. Không để lại `TODO`, code chết, hay hàm rỗng.

Chưa đủ 5 mục thì vẫn là **đang làm**, không được tính vào % tiến độ trong sheet kế hoạch.

## 7. TUYỆT ĐỐI KHÔNG COMMIT

| Không commit | Lý do |
|---|---|
| Hóa đơn/chứng từ **thật** chưa ẩn danh | Chứa MST, giá, tên đối tác của doanh nghiệp |
| `.env`, khóa API, token (kể cả khóa Gemini) | Lộ khóa là phải thu hồi ngay |
| Trọng số mô hình OCR, tệp nhị phân lớn | Phình repo, tải lại được |
| File `.xlsx` kế hoạch và danh sách của Khoa | Có thông tin cá nhân, một file ghi *lưu hành nội bộ* |
| `CLAUDE.md`, `QUY_DINH_DU_AN.md` | Tài liệu cá nhân, chỉ để ở máy |

Dữ liệu thật để **ngoài repo**. Chỉ commit dữ liệu tổng hợp và bộ sinh dữ liệu.
Luôn `git status` và đọc lại diff trước khi commit.

## 8. Nhịp tuần

1. Trong tuần: commit nhỏ, thường xuyên, push cuối ngày.
2. Cuối tuần: merge vào `dev`, kiểm tra hệ thống chạy được.
3. **Gửi email báo cáo cho thầy** — tiêu đề `[BC-DACNTT/KLTN/DATN-261] - 52300155 - 52300159`,
   reply vào đúng thread cũ, không tạo mail mới.
   Hai tuần liên tiếp không báo cáo ⇒ thầy có quyền từ chối hướng dẫn tiếp.
4. Hai tuần một lần: họp Google Meet chiều thứ Bảy sau 13h30.
