# Kế hoạch tuần 5 và giữa kỳ — 25/09/2026

Tuần 5: 21–27/09. Tuần 6: 28/09–04/10. Theo kế hoạch hiện có, hồ sơ giữa kỳ cần nộp ELIT **trước 30/09**, kèm xác nhận của GVHD. Mốc 43% trong bảng kế hoạch là mục tiêu tuần 5, chưa phải tỷ lệ hoàn thành đã kiểm chứng.

## Đang ở đâu

| Hạng mục | Trạng thái và bằng chứng |
|---|---|
| Chuẩn hóa dữ liệu | Có adapter cho 3 bộ; đã tải ảnh/nhãn SROIE và CORD-v2 ngày 25/09, xem [chuẩn bị receipt](RECEIPT_PREPARATION.md). Mapping SROIE mới được lưu rõ, chưa đối chiếu mapping cũ của Thịnh |
| GNN và RoBERTa | GNN/DocILE đã train đủ 20 epochs, test đủ 500 tài liệu: entity F1 66,82%, field-value F1 56,87%, đã đối chiếu hash và tái chấm giá trị trường. RoBERTa/DocILE cũng hoàn tất 20 epochs, best epoch 19: entity F1 79,69%, field-value F1 66,39%, đã kiểm tra hash và tái chấm trên cùng 500 test. GNN/SROIE và GNN/CORD đã kiểm chứng đầy đủ: entity F1 80,77% / 69,73%; bốn key SROIE gốc F1 54,70%. RoBERTa receipt full version 2 đã hoàn tất và kiểm chứng: SROIE entity F1 88,37%, bốn key gốc F1 73,60%; CORD entity F1 91,99%, field-value F1 94,08% |
| Độ tin cậy | Đã sửa băm GNN, nạp checkpoint, nhãn/trang khi chấm và metadata xuất dự đoán; đang chờ peer review/merge |
| LayoutLMv3 và LiLT | Chưa hiện thực |
| Hai VLM | Thịnh đề xuất Qwen2.5-VL và InternVL. Đề xuất thử 3B-Instruct / InternVL2.5-2B; chưa hiện thực hoặc fine-tune |
| Demo | Có khung web và danh sách tài liệu; chưa có luồng upload → suy luận → highlight hoàn chỉnh |
| Tích hợp | Nhánh dev còn ở PR #5; thay đổi của Thịnh và bản sửa phải qua PR, không tính là đã hoàn thành theo POLICY trước khi được duyệt/merge |

## Ưu tiên đến hết tuần 5

1. **24–25/09: khóa đường chạy đáng tin cậy.** Review/merge PR tích hợp, bàn giao checkpoint và đường dẫn SROIE/CORD, tạo lại export DocILE theo split 4.680/500/500. Không tái sử dụng train export cũ để học trước khi chấm test mới.
2. **25–27/09: tái chạy và chấm.** GNN và RoBERTa đã hoàn tất và kiểm chứng trên cả ba bộ. Tiếp theo chuẩn bị LayoutLMv3/LiLT và hai VLM theo cùng split/protocol; chưa có kết quả cho bốn mô hình này. Xuất đủ 500/100/63 tài liệu tương ứng. Lưu metrics sidecar và [bảng so sánh đã kiểm chứng](../reports/verified_comparison.md). Chạy DocILE evaluator cho số liệu mang nhãn chính thức.
3. **Trong tuần 5: chốt phạm vi sáu mô hình.** Xác nhận hai VLM, kích thước/checkpoint, tài nguyên chạy, cách prompt/giải mã và bộ dữ liệu bổ sung nếu còn ngoài ba bộ hiện có. Cập nhật lại các dòng kế hoạch đang ghi graph vision/Agent; không tự thay tên hai VLM.

## Chuẩn bị hồ sơ trước 30/09

- **27–28/09:** hoàn thiện demo tối thiểu trên mô hình đã kiểm chứng: đưa tài liệu vào, chạy suy luận, hiển thị trường/vị trí và lỗi rõ ràng. Nếu chỉ hỗ trợ mẫu có OCR sẵn, ghi đúng giới hạn; upload PDF bất kỳ cần thêm bước OCR.
- **28/09:** khóa nội dung giữa kỳ: bài toán, dữ liệu/splits, sáu mô hình trong phạm vi, kiến trúc đã làm, quy trình đánh giá, kết quả có bằng chứng và hạn chế. Bốn mô hình chưa hiện thực phải nằm ở phần công việc tiếp theo.
- **29/09:** hai thành viên rà số liệu và đóng góp, xin GVHD xác nhận, kiểm tra hồ sơ ELIT và nộp trước hạn theo kế hoạch. Đây là việc cần nhóm thực hiện; chưa có email gửi hoặc hồ sơ được nộp bởi lần sửa mã này.

## Cần bàn giao để tiếp tục

Kiểm chứng cục bộ ngày 24/09: **98 tests passed, 5 skipped** (các kiểm tra cần dữ liệu SROIE/CORD chưa có). GNN chạy GPU 1 epoch với 16 train / 4 val, nạp checkpoint ở tiến trình mới để suy luận và resume thêm 1 epoch thành công. Đây chỉ là smoke test, không xác nhận chất lượng mô hình. Export DocILE mới đã tạo và kiểm tra checksum: 4.680 train / 500 val / 500 test; lưu trong `.cache/week5-export/docile`.

- Checkpoint GNN/RoBERTa, config, seed và split thực sự đã dùng khi train; checkpoint GNN cũ chỉ để truy vết vì phải train lại.
- Đường dẫn hoặc bản dữ liệu SROIE và CORD đúng phiên bản.
- Tên hai VLM và xác nhận “thêm dataset” đã đủ ba bộ hiện tại hay còn bộ khác.

Workbook `52300155_52300159.xlsx` chưa được chỉnh trong lần này. Tài liệu này là kế hoạch cập nhật cho phạm vi đã xác nhận, không phải chứng nhận hoàn thành hay thay đổi hạn nộp của trường.
