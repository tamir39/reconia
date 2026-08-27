# Reconia

Hệ thống đối soát hóa đơn và chứng từ nhập xuất
*Development of an Invoice and Inventory Document Reconciliation System*

Đồ án môn Dự án CNTT — ĐA161 — HK1/2026-2027
Nhóm Best Stuffs: Phí Vương Tường Tâm (52300155) · Vũ Hưng Thịnh (52300159)
GVHD: Dung Cẩm Quang

## Chạy thử

Một lệnh duy nhất, dựng cả cơ sở dữ liệu lẫn ứng dụng và tự chạy migration:

```bash
cp .env.example .env
docker compose -p reconia up -d --build
```

Mở http://localhost:8000

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
