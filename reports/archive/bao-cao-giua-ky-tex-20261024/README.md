# Báo cáo giữa kỳ Reconia

Mở `main.tex` để biên dịch. Phần lý thuyết và triển khai được mở rộng ngày 25/10/2026 từ bài báo gốc và source lưu cùng các lượt chạy Kaggle; không thay số liệu thực nghiệm. Cách trình bày chuyển theo template TDTU; trích dẫn chuyển sang tác giả–năm (APA), hình/bảng đánh số theo chương.

## Sửa nội dung

- `coverpage.tex`: bìa, thành viên, giảng viên.
- `frontmatter.tex`: lời cảm ơn, lời cam đoan, tóm tắt, danh mục.
- `NoiDungChinh/Chuong-1.tex` đến `Chuong-5.tex`: năm chương.
- `NoiDungChinh/MoHinh/`: nền tảng, ba mô hình và so sánh; được nạp từ Chương 2.
- `tailieu.bib`: tài liệu tham khảo.
- `format.tex`: định dạng chung.
- `picture/`: logo và hai hình kiến trúc từ bài báo gốc.

## Biên dịch

Overleaf: tải bộ ZIP nguồn lên, chọn `main.tex`, compiler **XeLaTeX** và TeX Live 2025 hoặc mới hơn, rồi bấm Recompile. Biber sẽ tạo trích dẫn và danh mục tài liệu tham khảo. Khi dùng TeX cục bộ:

```text
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

XeLaTeX dùng Times New Roman khi có, hoặc TeX Gyre Termes. Nguồn cũng hỗ trợ pdfLaTeX như template gốc. Không cần thư viện hoặc runtime huấn luyện model để build báo cáo.

Bản PDF kèm theo được biên dịch trực tiếp từ nguồn TeX bằng Tectonic 0.17.0 và Biber 2.17, dùng Times New Roman. Đã kiểm tra 8 tài liệu tham khảo học thuật, hai hình kiến trúc từ bài báo gốc, chín dòng kết quả P/R/F1 và liên kết mục lục.

## Nguồn template

Dựa trên [bnquys/report-latex-tdtu](https://github.com/bnquys/report-latex-tdtu), commit `606240c30a5b06d6c1cb9029a9ee05e2775b7771`. Repository này ghi nhận nguồn [Master Thesis Template for TDTU version Vietnamese](https://www.overleaf.com/latex/templates/master-thesis-templete-for-tdtu-version-vietnamese/phrsbqgqvqch) của Nhut Truong - Huynh, được công bố theo CC BY 4.0.

Điều chỉnh cho Reconia: thay nội dung mẫu bằng báo cáo giữa kỳ, bỏ trang hội đồng và công bố của luận văn, thu khoảng cách bìa cho tên đề tài dài, giữ lề A4/cỡ chữ/giãn dòng theo template, thêm hỗ trợ XeLaTeX. Hai hình BROS và PICK được cắt trực tiếp từ PDF trong papers.zip người dùng cung cấp, có ghi nguồn dưới hình. Bài RoBERTa không có sơ đồ kiến trúc nên chỉ giữ mô tả văn bản. Hồ sơ nguồn hình và mã SHA-256 nằm trong picture/figure-sources.json. Notebook và gói dữ liệu của nhóm được chuyển sang mục nguồn thực nghiệm, không đưa vào danh mục học thuật.


## Rà soát ngày 24/10/2026

Bỏ cả ba sơ đồ tự tạo khỏi báo cáo. Hình BROS: Hong và cộng sự (2022), Hình 2, trang 10769 (trang PDF 3). Hình PICK: Yu và cộng sự (2020), Hình 3, trang PDF 4. Các hình gốc không mô tả chính xác mọi khác biệt của bản triển khai; chú thích dưới hình đã nêu rõ giới hạn này. Căn hai cột tên thành viên và chèn nguyên ảnh chữ ký từ báo cáo NLP giữa kỳ trong College theo yêu cầu người dùng. Giữ nguyên toàn bộ số liệu thực nghiệm.

Nguồn ảnh được trình bày bằng dòng chữ nhỏ căn trái dưới chú thích; phần giải thích được chuyển về đoạn văn thường. Hồ sơ nguồn chữ ký nằm trong `picture/signature-sources.json`.


## Nghiên cứu mô hình ngày 25/10/2026

Chương 2 mở rộng cơ chế RoBERTa, BROS và PICK với công thức, nguồn trích dẫn và giới hạn diễn giải. Hai hình gốc chuyển về Chương 2. Chương 3 đối chiếu cấu hình và thuật toán của source Kaggle đã chạy, gồm các khác biệt PICK ở Union/CRF, đặc trưng ảnh, khoảng cách trong graph loss và lexicon SROIE. ZIP nguồn phải bao gồm toàn bộ cây `NoiDungChinh/`, kể cả `MoHinh/`.


## Bản giản lược ngày 25/10/2026

Bản PDF hiện tại có 25 trang, thay bản 38 trang trước đó. Rút phần lý thuyết, bỏ công thức attention và đồ thị, giảm chi tiết mã nguồn; giữ ba mô hình, hai hình gốc, tám tài liệu tham khảo và toàn bộ 27 giá trị P/R/F1. Tên tác giả nằm trên chữ ký, có ngày tháng; caption dùng dấu hai chấm; nguồn hình dùng tác giả–năm. Bỏ mục liệt kê notebook khỏi nội dung báo cáo.

Đây là báo cáo giữa kỳ, chưa phải bản khóa luận hoàn chỉnh. Hướng dẫn 2284 quy định nội dung khóa luận/đồ án hoàn chỉnh tối thiểu 30 trang, tối đa 80 trang (không tính các phần được loại trừ). Chưa kiểm tra Turnitin, công cụ nhận diện AI hoặc NotebookLM; không có chứng nhận đạt các công cụ này.

Ảnh ELIT người dùng cung cấp yêu cầu trước khi nộp có đồng ý của GVHD và kế hoạch có chữ ký; gói nộp MSSV1_MSSV2.zip gồm baocao.pdf, kehoach.pdf, baocao - turnitin.pdf và source_code/readme.txt; source_code từ 2GB trở lên dùng link.txt trỏ source.zip. Hạn trong ảnh: 23:59 ngày 21/11/2026. Chưa tạo gói nộp vì thiếu kế hoạch có chữ ký và kết quả Turnitin.

Nguồn thực nghiệm: https://www.kaggle.com/code/vhngthnh/roberta-model ; https://www.kaggle.com/code/vhngthnh/bros-model ; https://www.kaggle.com/code/vhngthnh/pick-model . Notebook reconia-thinh-token-labels-20261024 dùng xuất nhãn dự đoán từ checkpoint để phân tích lỗi, không phải một mô hình mới và không huấn luyện lại.


## Chỉnh theo góp ý ngày 25/10/2026

Hai khối chữ ký đặt Tác giả 1/2 và hướng dẫn ký phía trên, họ tên phía dưới chữ ký. Trích dẫn dùng et al. và and; nguồn hình căn giữa theo dạng (Nguồn: (Tác giả et al., năm)). Bỏ năm dòng nguồn cho dữ liệu, cấu hình và kết quả do nhóm tự thực hiện; thông tin làm tròn được giữ trong tên bảng kết quả. Ba công thức Precision, Recall và F1 tăng khoảng cách hàng thêm 14pt. PDF đã biên dịch lại và kiểm tra các trang thay đổi.
