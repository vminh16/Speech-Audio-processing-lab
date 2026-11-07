# Lab 04 — Tổng hợp tiếng nói

**Nguyễn Văn Minh — 2351260673**

## Phương pháp

Kho dữ liệu có 58 word-unit, 16 từ, mono PCM16 16 kHz. Chuẩn hóa văn bản đọc từng chữ số của số thứ tự. Ghép nối cố định `_01` làm baseline; Viterbi tối thiểu hóa target cost (duration/RMS) và join cost (phổ/RMS), sau đó cross-fade 15 ms. Các hình 01–03 minh họa dữ liệu, biên nối và lựa chọn unit. Thí nghiệm cross-fade khảo sát 0, 5, 15, 30 ms; chưa có điểm nghe để khẳng định mức tối ưu chủ quan.

Prosody: kiểm chứng F0 bằng tự tương quan và YIN; biến đổi tốc độ 0,8/1/1,2× và pitch −2/0/+2 semitone. Mẫu gốc đại diện tốc độ 1× và pitch 0. Phase vocoder có thể sinh artifact; pitch shift bằng librosa không bảo đảm giữ nguyên formant. Các mô phỏng giọng trong audio_output là thí nghiệm phụ, không chứng minh chuyển giọng vùng miền.

**HMM:** trích F0 bằng YIN, F1–F3 từ nghiệm LPC và RMS theo frame 10 ms; huấn luyện Gaussian HMM trái–phải 3 trạng thái cho mỗi từ, covariance chéo, 40 vòng EM tối đa, seed 42. Đặc trưng gồm tĩnh và sai phân bậc một. Trạng thái được khởi tạo theo ba phần thời gian, không phải nhãn âm vị thật. Thời lượng được lấy trung bình từ các đường Viterbi của bản ghi. Sinh quỹ đạo giải hệ MLPG `WᵀU⁻¹W c = WᵀU⁻¹μ`, dùng trung bình/phương sai đã học cho cả tĩnh và delta; vocoder source–filter sinh waveform. Mô hình và tham số chuyển trạng thái được lưu tại `data/models/hmm.json`. Đã thử speed 1,2× và pitch 1,1×.

Hạn chế HMM: dữ liệu rất nhỏ, F0/YIN chưa phân loại hữu thanh/vô thanh đáng tin cậy, LPC có thể nhầm formant, mô hình word-level không thể khái quát ngoài từ vựng. Vocoder đơn giản có thể nghe rè; kết quả không được mặc định là phát âm rõ. Một số vòng EM dừng với mức giảm likelihood rất nhỏ do số học/ước lượng covariance; không khẳng định đã đạt cực đại toàn cục.

**Neural:** Piper/VITS `vi_VN-vais1000-medium`, phiên bản `v1.0.0`, thực sự nạp ONNX pretrained và suy luận CPU offline. Model card ghi 1 người nói, 22.050 Hz, fine-tune từ lessac; đầu ra được resample về 16 kHz để so sánh. Trọng số được tải một lần và kiểm SHA-256 trong manifest. Hình mel là phổ phân tích từ waveform vì backend không trả mel trung gian; không trình bày nó như latent của mô hình. VITS tích hợp tổng hợp, vì vậy không tách latency acoustic/vocoder giả định.

## Thí nghiệm và đánh giá

Cùng ba câu: “Mời khách hàng số 23 đến quầy 5”, “Mời khách hàng số 08 đến quầy 10”, “Mời khách hàng số 14 đến quầy 2”. [Bảng đo](evaluation/measurements.csv), [bảng tổng hợp](evaluation/summary.csv) và [biểu đồ](figures/05_methods_comparison.png) được xuất từ lần chạy thực tế. Đo latency từ text đến waveform, chưa gồm tính phổ minh họa hoặc ghi file. Bảng hiện tại được tạo trong pipeline sau giai đoạn 6, vì vậy neural đã nạp model vào cache; không bao gồm chi phí khởi động. Chạy `python -m src.run_phase6` trong tiến trình mới sẽ đo cả chi phí nạp ở câu đầu. Chỉ ba câu/lần, không coi đó là benchmark thống kê ổn định. Dung lượng là file dữ liệu/model, không phải RAM.

| Phương pháp | Latency trung bình (ms) | RTF | Dữ liệu/model (MiB) | MOS |
|---|---:|---:|---:|---|
| concat | 49.19 | 0.0030 | 3.184 | Chưa thu |
| hmm | 108.00 | 0.0198 | 0.038 | Chưa thu |
| neural | 82.97 | 0.0409 | 60.278 | Chưa thu |

Các mẫu nghe cùng RMS mục tiêu 0,08, giới hạn peak nên RMS thực có thể lệch. Bộ S1_A…S3_C có ánh xạ ẩn danh cố định, khóa dành riêng người tổ chức. Giao diện `evaluation/listening.html` thu nội dung nghe được, naturalness và continuity 1–5; bảng khóa và câu chuẩn phải được giấu khỏi người nghe. Intelligibility tính từ số từ đúng trên tổng số từ sau đối chiếu. **Chưa có người nghe thật: MOS, intelligibility và continuity để trống.** Chưa thể kết luận backend nào rõ/tự nhiên hơn. Cần thu 5–10 người nghe, tổng hợp và bổ sung kết luận để hoàn thành phần đánh giá chủ quan của SPEC.

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

## Chạy lại

Theo README: setup môi trường, tải model một lần, huấn luyện HMM (hoặc dùng model đã lưu), chạy pipeline và pytest. Audio, hình và số đo được commit; trọng số ONNX tải bằng script có checksum. Báo cáo Markdown thay PDF theo yêu cầu người dùng.

Nguồn: [Piper model card](https://huggingface.co/rhasspy/piper-voices/blob/v1.0.0/vi/vi_VN/vais1000/medium/MODEL_CARD), [Piper](https://github.com/OHF-Voice/piper1-gpl), [hmmlearn](https://hmmlearn.readthedocs.io/en/stable/).

Kiểm chứng lần cuối: 48 kiểm thử đạt; pipeline và benchmark chạy thành công. Phần chấm nghe vẫn chờ dữ liệu thực.
