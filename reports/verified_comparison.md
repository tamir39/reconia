# Kết quả đã kiểm chứng — 25/09/2026

## DocILE

| Mô hình | Train / Val / Test | Epoch tốt nhất | Val entity F1 | Test entity P / R / F1 | Test field-value F1 | Thiết bị |
|---|---|---:|---:|---|---:|---|
| GNN | 4.680 / 500 / 500 | 19 / 20 | 59,83% | 63,14% / 70,95% / **66,82%** | **56,87%** | Tesla T4 |
| RoBERTa + bố cục 2D | 4.680 / 500 / 500 | 19 / 20 | 74,76% | 77,46% / 82,05% / **79,69%** | **66,39%** | Tesla T4 |

GNN được lấy từ `gnn-docile-seed42-v1.zip` do nhóm tải từ Kaggle. Đã kiểm tra CRC của ZIP, hash checkpoint/predictions/split/manifest, đủ 500 ID test và không trùng train/test. Điểm field-value đã được tính lại độc lập từ ground truth cục bộ và khớp kết quả. Xem [bằng chứng](runs/gnn_docile_20260924/verification.json), [cấu hình](runs/gnn_docile_20260924/run-config.json), [metrics](predictions/predictions_gnn_docile_test.metrics.json).

Entity metric: `ocr-bio-exact-span-v2`. Field-value metric: `field-value-nfkc-casefold-v1`. Đây là test nội bộ lấy từ train gốc của DocILE, không phải test chính thức. Điểm của hai protocol không thể dùng thay thế nhau. Độ trễ GNN ghi nhận khoảng 95,1 ms/tài liệu, bao gồm các bước trong prediction pipeline; so sánh tốc độ cần cùng thiết bị và quy trình.

Các báo cáo mô hình/dataset khác chưa có provenance tương ứng vẫn là kết quả lịch sử. Bảng này chỉ dùng các lần chạy đã kiểm chứng, không dùng điểm smoke test. Đây là một seed khởi đầu, chưa phải kết luận thống kê qua nhiều lần chạy.

RoBERTa: ZIP `roberta-docile-seed42-v1.zip` tải ngày 25/09 chứa đủ 20 epochs và 500 dự đoán test. CRC mọi tệp, hash checkpoint/predictions/split/manifest và mã nguồn đã khớp; không có ID test trùng train hoặc val. Toàn bộ chỉ số field-value đã được tái tính từ nhãn DocILE gốc và khớp. Xem [bằng chứng](runs/roberta_docile_20260925/verification.json), [cấu hình](runs/roberta_docile_20260925/run-config.json), [metrics](predictions/predictions_roberta_docile_test.metrics.json). Lần kiểm tra cục bộ này không chạy lại suy luận; entity F1 lấy từ kết quả xuất của lần chạy đã kiểm tra provenance.

Trên cùng split và seed 42, RoBERTa cao hơn GNN **12,87 điểm phần trăm entity F1** và **9,52 điểm phần trăm field-value F1**. Độ trễ RoBERTa ghi nhận 94,6 ms/tài liệu; chưa đủ cơ sở kết luận khác biệt tốc độ có ý nghĩa. Phiên version 3 bị hủy là bản chạy trùng, không phải lỗi của kết quả version 2 này.

## SROIE và CORD

| Mô hình / bộ dữ liệu | Train / val / test | Best / tổng epochs | Val entity F1 | Test entity P / R / F1 | Giá trị đoạn OCR F1 | Bốn key SROIE gốc F1 |
|---|---|---:|---:|---|---:|---:|
| GNN / SROIE | 500 / 63 / 63 | 15 / 20 | 78,36% | 78,19% / 83,52% / **80,77%** | 81,32% | **54,70%** |
| GNN / CORD-v2 | 800 / 100 / 100 | 19 / 20 | 73,21% | 64,50% / 75,89% / **69,73%** | **70,46%** | — |
| RoBERTa / SROIE | 500 / 63 / 63 | 8 / 13 | 81,87% | 86,22% / 90,62% / **88,37%** | 89,20% | **73,60%** |
| RoBERTa / CORD-v2 | 800 / 100 / 100 | 9 / 14 | 92,91% | 91,71% / 92,27% / **91,99%** | **94,08%** | — |

Hai lần GNN chạy seed 42 trên Tesla T4. ZIP, hash checkpoint/predictions/mã nguồn,
manifest đối chiếu receipt và đúng toàn bộ ID test đều đã khớp; test không trùng
train/val. Tất cả chỉ số field-value từ đoạn OCR đã được tái tính từ nhãn cục bộ
và khớp. Entity F1 lấy từ output đã kiểm chứng provenance, không chạy lại suy luận.
Xem [SROIE](runs/gnn_sroie_20260925/verification.json) và
[CORD](runs/gnn_cord_20260925/verification.json).

SROIE dùng split nội bộ với mapping nguồn mới; không khẳng định giống mapping cũ
của Thịnh. Chỉ số `sroie-four-field-nfkc-casefold-v1` ghép mọi đoạn dự đoán cùng loại
theo thứ tự OCR, rồi so với bốn key gốc; 63 test có 251 giá trị được chú giải. Một
address thiếu nhãn được loại khỏi phép chấm. Đã đối chiếu riêng trực tiếp với các
file key gốc: 131 TP / 97 FP / 120 FN. Đây là phép chấm dành cho so sánh với đầu ra
VLM một giá trị mỗi key; không thay thế hoặc trộn với điểm đoạn OCR 81,32%.
[Metrics bốn key](predictions/gnn_sroie_test.four-field.metrics.json).

RoBERTa full version 2 đã hoàn tất và được kiểm chứng trên cùng split, seed 42 và Tesla T4. SROIE dừng sớm ở epoch 13, chọn best epoch 8; CORD dừng ở epoch 14, chọn best epoch 9, đúng patience 5 theo validation F1. Đã kiểm tra CRC và SHA-256 mọi tệp ZIP, hash checkpoint, predictions, mã nguồn, manifest đối chiếu receipt và đầy đủ ID test; không trùng train/val. Toàn bộ chỉ số field-value đã tái tính và khớp. Entity F1 lấy từ output, không chạy lại suy luận cục bộ. Xem [SROIE](runs/roberta_sroie_20260925/verification.json) và [CORD](runs/roberta_cord_20260925/verification.json).

RoBERTa đạt bốn key SROIE gốc F1 **73,60%**; đối chiếu trực tiếp key gốc cho **184 TP / 65 FP / 67 FN**, hỗ trợ 251 giá trị. [Metrics bốn key](predictions/roberta_sroie_test.four-field.metrics.json). So với GNN trên cùng split, tăng **18,90 điểm phần trăm** ở phép chấm này; CORD tăng **23,62 điểm phần trăm field-value F1**. Đây vẫn là kết quả một seed.

GNN và RoBERTa hiện có sáu lần chạy đầy đủ đã kiểm chứng trên ba bộ dữ liệu. LayoutLMv3, LiLT và hai VLM chưa có kết quả; không suy ra toàn bộ đề tài đã hoàn tất.
