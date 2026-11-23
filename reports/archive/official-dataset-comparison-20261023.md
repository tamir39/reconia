# Đối chiếu dữ liệu với nguồn phát hành — 23/10/2026

Phạm vi: xác minh bản raw của `vhngthnh/reconia-data`. Giữ nguyên dữ liệu nguồn, kể cả ảnh trùng, nhãn thiếu, text rỗng và metadata khác kích thước ảnh. Không làm sạch, sửa GT, thay split, chạy train hoặc chấm test.

| Bộ | Nguồn đối chiếu | Kết quả |
|---|---|---|
| CORD-v2 | NAVER CLOVA, `naver-clova-ix/cord-v2`, revision `7f0115a4b758a71d6473b8d085751692da2fef98` | Khớp toàn bộ 1.000 mẫu: 800 train, 100 validation, 100 test. ID, byte ảnh và toàn bộ annotation đều khớp. |
| WildReceipt | Gói `wildreceipt.tar` của OpenMMLab, được dẫn trong repository MMOCR | Cả 1.773 file khớp từng byte, không thiếu/thừa file so với gói phát hành. Danh sách và nội dung annotation của 1.267 train, 472 test khớp cả thứ tự. |
| SROIE | Thư mục Drive gốc được dẫn trong thông báo năm 2019; GT test đối chiếu thêm với `urbikn/sroie-datasetv2`, version 4 | Khớp từng byte 2.572 file với Drive và đủ 347 GT test với bản Kaggle tham chiếu. Đủ ID của 626 train và 347 test. Chưa tải được gói GT test trực tiếp từ RRC vì cổng này timeout. |

## Cách xác minh

- CORD: lấy revision và SHA-256 của sáu file Parquet từ API repository chính thức. Sáu file đã cache đều khớp mã băm công bố. Đọc toàn bộ mẫu từ các file này và so với bản Arrow raw hiện tại theo split và ID; so byte ảnh và toàn bộ JSON `ground_truth`, không chỉ số mẫu hay các trường dùng để chấm điểm.
- WildReceipt: đọc trực tiếp gói TAR từ máy chủ OpenMMLab theo luồng; so từng file với bản raw. Không giải nén đè lên dữ liệu. Đối chiếu bổ sung toàn bộ bản ghi trong `train.txt` và `test.txt`.
- SROIE: tải bản tham chiếu vào thư mục cache riêng, đối chiếu tên/ID rồi so toàn bộ byte của từng file. Thư mục Drive có nhiều bản sao cùng tên; mỗi file hiện tại đều tìm được bản tương ứng khớp byte. GT test đối chiếu với bản Kaggle version 4. Hash trước và sau xác nhận toàn bộ 2.919 file raw SROIE và file prepared test hiện tại không đổi.

## Phạm vi xác minh SROIE

| Thành phần | Train | Test | Kết quả |
|---|---:|---:|---|
| Ảnh | 626 | 347 | Tất cả khớp từng byte |
| Text và bounding box | 626 | 347 | Tất cả khớp từng byte |
| GT trích xuất: company, date, address, total | 626 | 347 | 626 GT train khớp Drive; 347 GT test khớp Kaggle version 4, chưa đối chiếu trực tiếp với RRC |

Gói test của Task 1/2 có 361 file text/box, còn gói Task 3 có 347 ảnh. Toàn bộ 347 ID test hiện tại đều có trong nguồn; chênh lệch 14 file thuộc hai gói tác vụ khác nhau, không phải thiếu 14 mẫu trong bản Task 3 hiện tại.

Link Drive xuất hiện trong thông báo SROIE lưu từ năm 2019 và được một repository tham dự cùng thời dẫn là “Original dataset”. Tuy nhiên, chưa xác minh lại được đường dẫn trực tiếp từ cổng RRC đang không truy cập được, cũng chưa xác thực độc lập chủ tài khoản đăng thông báo. Vì vậy, kết luận chính xác là **khớp bản Drive gốc có bằng chứng lịch sử**, chưa khẳng định toàn bộ dữ liệu đã được xác minh trực tiếp với cổng chính thức. Cả thư mục Drive và danh sách gói Baidu cùng thông báo đều không cung cấp GT test Task 3; không dùng bản bên thứ ba đã sửa nhãn để thay thế.

Các đặc điểm đã ghi nhận ở CORD và WildReceipt cũng tồn tại trong nguồn phát hành. Ví dụ, 30 file WildReceipt nằm ngoài danh sách train/test là file có sẵn trong gói OpenMMLab; không tự thêm chúng thành mẫu huấn luyện và không xóa chúng.

## Nguồn của 347 GT test SROIE hiện tại

**Nguồn trực tiếp là dataset Kaggle `vhngthnh/reconia-data` của Thịnh.** Các file nằm ở `sroie/test/key/*.txt`. Kiểm tra trước đó đã xác nhận bản raw khớp gói tải Kaggle; lần này xác nhận cả 347 GT trong `.cache/sroie/test.jsonl` giữ đúng nội dung từ các file raw tương ứng. Chúng không do bước PaddleOCR của dự án tạo ra.

Trong commit `1ceed2e` ngày 05/10/2026, code chuẩn bị dữ liệu đã có thao tác chép `SROIE_test_gt_task_3/<id>.txt` sang `test/key/<id>.txt` bằng `shutil.copy2`. Thao tác này được chuyển sang `tools/setup_data.py` ở commit `dd115f7`. Code chỉ chép file, không sinh đáp án; không ghi URL tải thư mục ban đầu.

Tìm được đường dẫn `https://rrc.cvc.uab.es/downloads/SROIE_test_gt_task_3.zip` trong hướng dẫn tải test của ViBERTgrid, dẫn về cổng RRC. Tên gói trùng thư mục đầu vào của code cũ. Đây là gói GT riêng, không có trong thư mục Drive đã kiểm tra. Tuy nhiên, tải trực tiếp gói RRC hiện vẫn timeout và không tìm thấy bản lưu ZIP trong truy vấn Internet Archive đã thực hiện.

Đã hoàn tất đối chiếu với `urbikn/sroie-datasetv2`, version 4: **347/347 tên file, kích thước và nội dung đều khớp từng byte**. Sau khi hết thời gian giới hạn HTTP 429, đã dùng lại 71 file tham chiếu cũ và tải riêng 276 file còn lại theo thứ tự; lượt tiếp tục không gặp lỗi tải hoặc giới hạn mới. Người đăng mô tả đây là bản sắp xếp lại dữ liệu của ban tổ chức; sự trùng khớp không chứng minh Thịnh đã tải từ chính mirror này.

**Kết luận:** truy được nguồn trực tiếp, cách đưa GT vào dự án và xác nhận cả 347 GT test khớp bản Kaggle tham chiếu. Chưa đủ bằng chứng để khẳng định URL Thịnh đã tải ban đầu là RRC hay một mirror. Không có căn cứ từ lượt kiểm tra này để coi GT hiện tại là tự tạo hoặc cần sửa. Cả 2.919 file raw SROIE và prepared test vẫn khớp hash trước khi kiểm tra; không sửa lỗi, làm sạch hoặc loại mẫu.

## Nguồn và bằng chứng

- CORD: https://github.com/clovaai/cord ; https://huggingface.co/datasets/naver-clova-ix/cord-v2/tree/7f0115a4b758a71d6473b8d085751692da2fef98
- WildReceipt: https://github.com/open-mmlab/mmocr/blob/main/dataset_zoo/wildreceipt/metafile.yml ; https://download.openmmlab.com/mmocr/data/wildreceipt.tar
- SROIE: https://rrc.cvc.uab.es/?ch=13&com=downloads
- Bản Drive gốc SROIE đã đối chiếu: https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2
- Thông báo SROIE năm 2019 chứa link Drive/Baidu: https://github.com/SYZNKJ2019/SROIE2019/blob/b0e71e636de5574d36a6b953ee6977eb798312c0/README.md
- Dẫn chứng cùng thời về link “Original dataset”: https://github.com/zzzDavid/ICDAR-2019-SROIE#data
- Bản lưu trang SROIE chính thức: https://web.archive.org/web/20230101051712/https://rrc.cvc.uab.es/?ch=13&com=downloads
- Nguồn trực tiếp của dự án: https://www.kaggle.com/datasets/vhngthnh/reconia-data
- Code chép GT cũ: https://github.com/tamir39/reconia/blob/dd115f7528478307c59aa15f48c02b674aa984d4/tools/setup_data.py
- Hướng dẫn dẫn đến gói GT test riêng của RRC: https://github.com/ZeningLin/ViBERTgrid-PyTorch/blob/main/readme.md#21-icdar-sroie
- Bản Kaggle tham chiếu GT: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/versions/4
- Chi tiết máy đọc trong `.cache/official-dataset-comparison-20261023/`: `cord-comparison.json`, `wild-comparison.json`, `sroie-drive-comparison.json`, `sroie-drive-files.jsonl`, `sroie-drive-inventory-comparison.json`, `sroie-source-evidence.json`.
- Bằng chứng truy nguồn GT test: `sroie-test-gt-provenance.json`, `sroie-test-gt-kaggle-comparison.json`, `sroie-test-gt-kaggle-files.json`, `kaggle-test-gt-download-error.json` trong cùng thư mục cache trên.

Không sửa code xử lý, dataset Kaggle, raw, prepared, OCR cache hoặc GT trong lượt đối chiếu này.
