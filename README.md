# Reconia

Hệ thống đối soát hóa đơn và chứng từ nhập xuất
*Development of an Invoice and Inventory Document Reconciliation System*

Đồ án môn Dự án CNTT — ĐA161 — HK1/2026-2027
Nhóm Best Stuffs: Phí Vương Tường Tâm (52300155) · Vũ Hưng Thịnh (52300159)
GVHD: Dung Cẩm Quang

## Chạy thử

Trên Windows, chạy tệp `run.bat`. Tệp này kiểm tra Docker, kiểm tra xung đột cổng, dựng
cơ sở dữ liệu và ứng dụng, chạy migration, nạp dữ liệu minh họa và in ra các địa chỉ truy cập.

```
run.bat
```

Hoặc chạy trực tiếp bằng Docker Compose:

```bash
cp .env.example .env
docker compose -p reconia up -d --build
docker compose -p reconia exec app python -m app.seed
```

Mở http://localhost:8010

**Cổng sử dụng:** ứng dụng dùng cổng `8010`, PostgreSQL dùng cổng `5437`. Các cổng này được
chọn để tránh xung đột với những dự án khác trên cùng máy. Sửa trong `run.bat` nếu cần.

Dừng: `docker compose -p reconia down` (thêm `-v` để xóa luôn dữ liệu).

<details>
<summary>Chạy trực tiếp không qua Docker (khi phát triển)</summary>

```bash
docker compose -p reconia up -d db
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m alembic upgrade head
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```
</details>

## Kiểm thử

```bash
./.venv/Scripts/python.exe -m pytest -q
```

Bộ kiểm thử không yêu cầu Docker; các kiểm thử sử dụng SQLite trong bộ nhớ.

## Cấu trúc

```
app/
  models/       Lược đồ CSDL (nguồn sự thật của ERD)
  intake/       Module 1 — tiếp nhận, phân loại, tiền xử lý
  extraction/   Module 2 — trích xuất, chuẩn hóa
  matching/     Module 3 — đối soát, ghép nối
  rules/        Module 4 — ràng buộc, phát hiện bất thường
  agent/        Module 5 — Agent điều phối, rà soát của con người
  audit/        Module 6 — nhật ký, bộ nhớ hiệu chỉnh
  ui/           Giao diện (Jinja2 + HTMX)
run.bat         Khởi chạy toàn hệ thống trên Windows
run-local.ps1   Kịch bản khởi chạy được run.bat gọi
docs/
  SPEC.md       Đặc tả: 6 module, 5 nhóm kết quả, 12 quy tắc
  AGENT.md      Kiến trúc Agent điều phối (Module 5)
  ERD.md        Sơ đồ quan hệ
```

## Tài liệu

- [POLICY.md](POLICY.md) — quy ước nhánh, commit, PR, code
- [docs/SPEC.md](docs/SPEC.md) — đặc tả nghiệp vụ
- [docs/AGENT.md](docs/AGENT.md) — kiến trúc Agent
- [docs/ERD.md](docs/ERD.md) — lược đồ dữ liệu
