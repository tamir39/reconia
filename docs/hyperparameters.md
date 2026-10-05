# Cấu Hình Siêu Tham Số Huấn Luyện

## 1. Tham số cố định toàn bộ mô hình
- Kích thước batch hiệu dụng: 16
- Bộ tối ưu hóa: AdamW
- Suy giảm trọng số: 0.01 (loại trừ bias và LayerNorm)
- Hạt giống ngẫu nhiên: 42
- Số epoch tối đa: 20
- Ngưỡng dừng sớm (patience): 5

## 2. Bảng tham số theo mô hình

| Mô hình | Tốc độ học (`lr`) | `batch_size` | `accum_steps` | Batch hiệu dụng | `epochs` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `roberta` | `3e-5` | `8` | `2` | 16 | 20 |
| `bros` | `3e-5` | `8` | `2` | 16 | 20 |
| `layoutlmv3` | `3e-5` | `4` | `4` | 16 | 20 |
| `donut` | `3e-5` | `2` | `8` | 16 | 20 |
| `pick` | `1e-4` | `4` | `4` | 16 | 20 |
| `qwen2_5_vl` | `1e-4` | `1` | `16` | 16 | 20 |
