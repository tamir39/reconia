# Kết quả đã kiểm chứng — DocILE, 25/09/2026

| Mô hình | Train / Val / Test | Epoch tốt nhất | Val entity F1 | Test entity P / R / F1 | Test field-value F1 | Thiết bị |
|---|---|---:|---:|---|---:|---|
| GNN | 4.680 / 500 / 500 | 19 / 20 | 59,83% | 63,14% / 70,95% / **66,82%** | **56,87%** | Tesla T4 |
| RoBERTa + bố cục 2D | 4.680 / 500 / 500 | 19 / 20 | 74,76% | 77,46% / 82,05% / **79,69%** | **66,39%** | Tesla T4 |

GNN được lấy từ `gnn-docile-seed42-v1.zip` do nhóm tải từ Kaggle. Đã kiểm tra CRC của ZIP, hash checkpoint/predictions/split/manifest, đủ 500 ID test và không trùng train/test. Điểm field-value đã được tính lại độc lập từ ground truth cục bộ và khớp kết quả. Xem [bằng chứng](runs/gnn_docile_20260924/verification.json), [cấu hình](runs/gnn_docile_20260924/run-config.json), [metrics](predictions/predictions_gnn_docile_test.metrics.json).

Entity metric: `ocr-bio-exact-span-v2`. Field-value metric: `field-value-nfkc-casefold-v1`. Đây là test nội bộ lấy từ train gốc của DocILE, không phải test chính thức. Điểm của hai protocol không thể dùng thay thế nhau. Độ trễ GNN ghi nhận khoảng 95,1 ms/tài liệu, bao gồm các bước trong prediction pipeline; so sánh tốc độ cần cùng thiết bị và quy trình.

Các báo cáo mô hình/dataset khác chưa có provenance tương ứng vẫn là kết quả lịch sử. Bảng này chỉ dùng các lần chạy đã kiểm chứng, không dùng điểm smoke test. Đây là một seed khởi đầu, chưa phải kết luận thống kê qua nhiều lần chạy.

RoBERTa: ZIP `roberta-docile-seed42-v1.zip` tải ngày 25/09 chứa đủ 20 epochs và 500 dự đoán test. CRC mọi tệp, hash checkpoint/predictions/split/manifest và mã nguồn đã khớp; không có ID test trùng train hoặc val. Toàn bộ chỉ số field-value đã được tái tính từ nhãn DocILE gốc và khớp. Xem [bằng chứng](runs/roberta_docile_20260925/verification.json), [cấu hình](runs/roberta_docile_20260925/run-config.json), [metrics](predictions/predictions_roberta_docile_test.metrics.json). Lần kiểm tra cục bộ này không chạy lại suy luận; entity F1 lấy từ kết quả xuất của lần chạy đã kiểm tra provenance.

Trên cùng split và seed 42, RoBERTa cao hơn GNN **12,87 điểm phần trăm entity F1** và **9,52 điểm phần trăm field-value F1**. Độ trễ RoBERTa ghi nhận 94,6 ms/tài liệu; chưa đủ cơ sở kết luận khác biệt tốc độ có ý nghĩa. Phiên version 3 bị hủy là bản chạy trùng, không phải lỗi của kết quả version 2 này.
