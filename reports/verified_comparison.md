# Kết quả đã kiểm chứng — DocILE, 24/09/2026

| Mô hình | Train / Val / Test | Epoch tốt nhất | Val entity F1 | Test entity P / R / F1 | Test field-value F1 | Thiết bị |
|---|---|---:|---:|---|---:|---|
| GNN | 4.680 / 500 / 500 | 19 / 20 | 59,83% | 63,14% / 70,95% / **66,82%** | **56,87%** | Tesla T4 |
| RoBERTa + bố cục 2D | Cùng split | Đang train | Chưa có | Chờ kết quả | Chưa có | Tesla T4 |

GNN được lấy từ `gnn-docile-seed42-v1.zip` do nhóm tải từ Kaggle. Đã kiểm tra CRC của ZIP, hash checkpoint/predictions/split/manifest, đủ 500 ID test và không trùng train/test. Điểm field-value đã được tính lại độc lập từ ground truth cục bộ và khớp kết quả. Xem [bằng chứng](runs/gnn_docile_20260924/verification.json), [cấu hình](runs/gnn_docile_20260924/run-config.json), [metrics](predictions/predictions_gnn_docile_test.metrics.json).

Entity metric: `ocr-bio-exact-span-v2`. Field-value metric: `field-value-nfkc-casefold-v1`. Đây là test nội bộ lấy từ train gốc của DocILE, không phải test chính thức. Điểm của hai protocol không thể dùng thay thế nhau. Độ trễ GNN ghi nhận khoảng 95,1 ms/tài liệu, bao gồm các bước trong prediction pipeline; so sánh tốc độ cần cùng thiết bị và quy trình.

Các báo cáo mô hình/dataset khác chưa có provenance tương ứng vẫn là kết quả lịch sử. Bảng này chỉ dùng các lần chạy đã kiểm chứng, không dùng điểm smoke test. Đây là một seed khởi đầu, chưa phải kết luận thống kê qua nhiều lần chạy.

RoBERTa: Kaggle version 1 đã hoàn tất smoke train/reload/inference; version 2 đang chạy đầy đủ với MODE: TRAIN và 4.680 tài liệu train (24/09). Chưa có điểm benchmark mới để so sánh.
