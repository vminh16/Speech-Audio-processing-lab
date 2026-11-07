# Lab 04 — Tổng hợp tiếng nói

**Sinh viên:** Nguyễn Văn Minh — 2351260673  
**Phạm vi:** thông báo xếp hàng tiếng Việt, 16 kHz, mono. Báo cáo này phân biệt kết quả đã đo với các yêu cầu chưa hoàn thành.

## Phương pháp và kết quả

Kho dữ liệu có 58 word-unit của 16 từ; văn bản được chuẩn hóa và chữ số được đọc từng chữ số. Ghép nối baseline dùng candidate `_01`; unit selection tối thiểu hóa tổng target cost và join cost bằng quy hoạch động, sau đó cross-fade 15 ms. Các hình `01`–`03` minh họa inventory, biên nối và lựa chọn unit. Thí nghiệm cross-fade khảo sát 0, 5, 15, 30 ms; chưa có đánh giá nghe để khẳng định mức nào tốt nhất.

Prosody: từ `ba_01`, chu kỳ tự tương quan 93 mẫu cho F0 = 16000/93 ≈ 172,04 Hz; YIN cho khoảng 171,34 Hz theo lần chạy được ghi trong ROADMAP. Đã tạo mẫu tốc độ 0,8× và 1,2×, pitch −2 và +2 semitone; mẫu gốc đại diện mức 1×/0 semitone. Các biến đổi này chưa được đánh giá nghe; không có bằng chứng đầy đủ rằng formant được bảo toàn hoàn toàn.

**Giới hạn giai đoạn 5:** `hmm_tts.py` mô phỏng các trạng thái bằng tham số F0/formant/thời lượng đặt tay, làm mượt bằng phạt sai phân và tổng hợp source–filter. Có thể quan sát các quỹ đạo và điều khiển pitch/tốc độ, nhưng đây chưa phải HMM được huấn luyện, và phép làm mượt không phải MLPG đầy đủ với phân bố tĩnh/động học từ dữ liệu.

**Giới hạn giai đoạn 6:** `neural_tts.py` tạo mel bằng quy tắc và dùng Griffin–Lim; không có mạng neural hay trọng số pretrained. Do đó các file mang tên `neural_tts` chỉ là mẫu mô phỏng, không đáp ứng yêu cầu Neural TTS offline trong SPEC.

Bộ [evaluation](evaluation/) có 9 mẫu cùng ba câu, chuẩn hóa RMS mục tiêu 0,08, và [số đo latency/RTF](evaluation/measurements.csv) cho ba cách hiện có. Latency gồm thời gian xử lý trên máy chạy và có thể thay đổi giữa các lần chạy; phép so sánh hai mô phỏng không đại diện cho HMM hay Neural TTS thực. Audio được giới hạn đỉnh sau chuẩn hóa nên RMS thực có thể khác đôi chút.

Chưa có người nghe thật, do đó **chưa có MOS, intelligibility hay kết luận về độ tự nhiên**. [Mẫu CSV](evaluation/ratings_template.csv) để nhập kết quả thật: mỗi dòng là một người, câu và phương pháp; điểm naturalness/continuity từ 1–5; intelligibility là số từ nghe đúng trên tổng số từ. Khi tổ chức nghe mù, đổi tên ngẫu nhiên các file và giữ bảng ánh xạ riêng khỏi người nghe; nhập lại tên phương pháp sau khi thu kết quả. Có thể dùng `src.evaluation.summarize_ratings` để tổng hợp.

## Trả lời câu hỏi

1. Các đoạn thu riêng có thể khác ngữ điệu, cao độ, âm lượng và phổ tại biên, khiến câu ghép kém tự nhiên.
2. Target cost đo độ phù hợp với ngữ cảnh đích; join cost đo độ tương thích giữa các unit liền kề. Chỉ tối ưu join có thể chọn unit nối êm nhưng không phù hợp thời lượng hoặc ngữ điệu.
3. Cross-fade dài chồng lấp phụ âm và nguyên âm, làm mờ chuyển tiếp và giảm độ rõ.
4. F0 thuộc nguồn kích thích; formant chủ yếu do cộng hưởng đường thanh quản, nên thay F0 không nhất thiết đổi formant.
5. PSOLA đặt lại khoảng cách các đoạn ngắn quanh pitch mark để đổi F0. Âm xát có thành phần nhiễu, dễ bị lặp giả tuần hoàn và rè khi kéo dài.
6. Ràng buộc delta đưa quan hệ giữa các frame vào tối ưu, giúp giảm bước nhảy bất hợp lý giữa trạng thái.
7. Cần nhiều bản ghi để bao phủ từ, ngữ cảnh, giọng và ngữ điệu khác nhau mà vẫn tìm được biên nối phù hợp.
8. Ghép nối chọn và nối waveform có sẵn; neural TTS dự đoán biểu diễn âm học/waveform từ mô hình học tham số. Chúng khác nhau về dữ liệu, khả năng khái quát và dạng lỗi.
9. Model tải ngẫu nhiên có thể khác ngôn ngữ, phiên bản, điều kiện chạy hoặc giấy phép, khiến kết quả khó tái lập và so sánh không công bằng.
10. Giọng tự nhiên nhưng khó hiểu có thể dùng cho âm thanh nền không mang thông tin hoặc nghệ thuật; không phù hợp thông báo xếp hàng cần nghe rõ.

## Chạy lại và việc còn thiếu

Từ thư mục Lab 04, cài `requirements.txt`, chạy `bash scripts/run_pipeline.sh`, `bash scripts/benchmark.sh` và `pytest -q`. Pipeline sinh mẫu mô phỏng và tập đánh giá; không tự tải pretrained model. Để đáp ứng đầy đủ SPEC cần backend HMM thật hoặc huấn luyện có cơ sở, mô hình neural tiếng Việt pretrained chạy offline, đánh giá nghe thật và so sánh lại ba phương pháp. Không nên nộp các mẫu mô phỏng như kết quả của hai mô hình đó.
