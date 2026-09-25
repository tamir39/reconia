# DocILE — ảnh trang cho LayoutLMv3

## Dữ liệu và quy trình

Dùng đúng 5.680 tài liệu của export OCR đã kiểm chứng: 4.680 train, 500 val,
500 test nội bộ. Tổng cộng 7.394 trang: 6.109 train, 635 val và 650 test.
Có 1.419 tài liệu nhiều trang. Không tạo lại split hoặc chạy OCR mới.

Đã render và kiểm chứng đủ 7.394 trang ngày 25/09/2026; 133 tests passed.
[Bằng chứng chuẩn bị](../reports/data/docile_images_20260925.json).
Bundle ảnh version 1 đã upload và xác nhận private. Smoke GPU version 1 đã qua train/reload/inference; source, manifest ảnh, hash predictions, ID val và field-value đã kiểm chứng cục bộ. Full version 2 đang chạy trên Tesla T4 với đầy đủ 4.680 train / 500 val / 500 test. Chưa có kết quả full run. Xem [bằng chứng Kaggle](../reports/data/layoutlmv3_docile_kaggle_20260925.json).

Không có toàn bộ tài liệu render giống hệt nhau giữa các split. Có **6 nhóm trang
giống hệt nhau qua biên split**: năm nhóm không có trường được chú giải; một mẫu
trang lặp có `vendor_name`, `vendor_address`, `payment_terms` trên 31 train,
2 val và 3 test (77/5/6 trường được chú giải tương ứng). Đây là overlap có sẵn
trong split nội bộ, ảnh hưởng cả baseline trước; không khẳng định split độc lập
ở mức nội dung trang. Giữ split hiện tại để so sánh công bằng; benchmark loại bỏ
trang/template trùng sẽ là thực nghiệm riêng, cần tái chạy các baseline cùng split mới.

[DocILE upstream](https://github.com/rossumai/docile/blob/main/docile/dataset/document_images.py)
render PDF bằng Poppler qua `pdf2image`. Pipeline này cũng render **toàn trang**
bằng Poppler, giữ hướng trang và toàn bộ nội dung, thay cho cách lấy ảnh nhúng
đầu tiên. PDF lỗi, trang không tồn tại hoặc thiếu ảnh đều báo lỗi; không thay bằng ảnh trắng.

Policy `poppler-full-page-rgb-png-longest1600-v1`: RGB PNG lossless, cạnh dài
1.600 pixel, giữ tỷ lệ trang. Đây là độ phân giải chuẩn bị ảnh; processor
LayoutLMv3 tiếp tục resize về kích thước backbone. Không dùng ảnh QA có vẽ bbox để train.

Mỗi tài liệu được kiểm tra:

- Nội dung OCR/nhãn từ adapter gốc khớp export sau phép làm tròn của export.
- Số trang và page index khớp giữa PDF render, OCR và annotation.
- Tỷ lệ trang render khớp kích thước annotation ở 200 DPI và OCR, sai số tối đa 0,5% do làm tròn pixel.
- Không có ảnh một màu khi trang có từ OCR.
- SHA-256 của PDF, OCR, annotation và từng PNG được lưu; manifest liên kết ảnh với nội dung OCR, trang và split cụ thể.

`PageImageStore` kiểm tra toàn bộ hash ảnh, ID/split và chữ ký tài liệu trước khi
đưa ảnh vào train/inference. `DOCILE_IMAGE_DIR` chỉ bổ sung ảnh cho export;
`DOCILE_EXPORT_DIR` tiếp tục cung cấp đúng từ/bbox/nhãn đã dùng cho GNN/RoBERTa.

## Chuẩn bị cục bộ

Cần Poppler (`pdftoppm`) trên PATH hoặc đường dẫn qua `--pdftoppm`.
`DocileAdapter.get_image()` trực tiếp cũng nhận biến `PDFTOPPM` nếu không có trên PATH.
Giữ ảnh, raw dataset và bundle ngoài Git.

```bash
python -m scripts.prepare_docile_images --root /data/docile/annotated-trainval --export /data/docile-export --output /data/docile-pages-v1 --pdftoppm /usr/bin/pdftoppm --workers 8
python -m scripts.prepare_layoutlmv3_docile --export /data/docile-export --images /data/docile-pages-v1 --output /data/kaggle-docile-images --owner YOUR_KAGGLE_USERNAME
```

Có thể chạy lại lệnh render sau khi bị ngắt: chỉ tái sử dụng record khi hash
nguồn, policy và mọi ảnh đã lưu còn khớp. Manifest đầy đủ chỉ được ghi sau khi
tất cả tài liệu hoàn tất. Bundle mới yêu cầu thư mục trống, gồm source allowlist,
OCR export và ảnh đã kiểm chứng; không có credential, PDF gốc hoặc model weights.

## Kaggle

[Notebook DocILE](../notebooks/kaggle_layoutlmv3_docile.ipynb) mặc định
`RUN_MODE = "smoke"`: 8 train / 4 val, một epoch, rồi reload checkpoint ở tiến trình mới.
Chỉ chuyển sang full sau khi smoke GPU và output đã qua kiểm tra.
Full giữ seed 42, tối đa 20 epochs, patience 5, batch 2 × accumulation 8 và
backbone/revision giống các run receipt. Chưa hỗ trợ resume.

Notebook kiểm tra checksum archive trước giải nén và manifest ảnh/nhãn trước
training; output ghi hash manifest ảnh, render policy và provenance mã nguồn.
Đây là đánh giá trên test nội bộ, không phải evaluator DocILE chính thức.

## Kiểm thử

Đã kiểm tra render PDF vector nhiều vùng màu (không có ảnh nhúng), trang xoay,
trang thiếu/sai chỉ số; bundle sai OCR/split/số trang/đường dẫn hoặc ảnh hỏng bị từ chối.
Tiny LayoutLMv3 chạy hết train → reload → test cho cả receipt và DocILE.
Kiểm tra trực quan mẫu thực tế bằng overlay OCR, gồm tài liệu nhiều trang;
chỉ sample ảnh được xem bằng mắt, không khẳng định đã xem trực quan mọi trang.

Người dùng đã cho phép riêng việc upload ảnh, notebook private, smoke rồi full nếu smoke qua. Dataset và notebook private đã được đọc lại để xác nhận; không public hoặc gửi email.
