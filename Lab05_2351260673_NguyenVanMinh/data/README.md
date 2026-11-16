# Dữ liệu Lab 5

## Phần 1 — Khử nhiễu (giữ nguyên)

- `clean/`: 7 clip FSDD đã ghép, gồm 6 dev (3 người × 2) để chọn cấu hình SS và `spk01_test_01.wav` làm clean reference/test.
- `noise/`: rain.wav và vacuum_cleaner.wav, mỗi clip 5 giây từ ESC-50.
- `raw/fsdd/`: chỉ giữ 70 bản ghi gốc tạo nên 7 clip phần 1, cùng README nguồn.
- `raw/esc50/`: noise gốc, metadata và giấy phép.
- `manifest.json`: 9 đầu vào phần 1 và nguồn gốc.

Các file dùng cho phần 1 được giữ nguyên byte; notebook `notebooks/Lab5.ipynb` và kết quả khử nhiễu vẫn dùng các đường dẫn cũ. Tạo lại bằng `python3 tools/prepare_data.py` (ffmpeg, Internet). Script này chỉ tạo dữ liệu phần 1, không tạo enrollment/ICA cũ nữa.

Nguồn: FSDD v1.0.10, https://github.com/Jakobovski/free-spoken-digit-dataset/tree/v1.0.10 (CC BY-SA 4.0); ESC-50, https://github.com/karolpiczak/ESC-50 (xem `raw/esc50/LICENSE`, CC BY-NC 3.0 cho bộ dữ liệu và ghi công nguồn clip). FSDD gốc 8 kHz được ghép/cắt/resample lên 16 kHz, không phải hội thoại tự nhiên.

## Phần 2 — Nhận dạng/xác thực người nói (mới)

Đặt tại `speaker_recognition/ami/`, từ cuộc họp tự nhiên AMI EN2009d:

- `enroll/`: 12 file = 3 người × 4.
- `dev/`: 6 file = 3 người × 2.
- `test/`: 6 file = 3 người × 2.
- `conversation_23s.wav`: đoạn hội thoại liên tục 23,28 giây để nghe, không đưa vào enrollment/dev/test.
- `metadata/`: annotation thủ công, mapping người nói, thông tin nguồn, giấy phép và header WAV gốc.
- `manifest.json`: speaker, split, thời gian nguồn, transcript và SHA256 từng clip.

Các clip dài 2,33–3,87 giây, mono 16 kHz PCM16; chỉ cắt liên tục theo ranh giới từ, không ghép chữ/câu, không thêm nhiễu, không khử nhiễu hay normalize. Xem README trong thư mục AMI để biết quy tắc chia và giới hạn.

Tải lại: `python3 tools/prepare_ami.py` (Python chuẩn, Internet). Script chỉ tải các byte WAV cần dùng, không tải toàn cuộc họp. Nó tải annotation chính thức để chọn clip, chỉ giữ phần annotation liên quan trong dự án.

Dữ liệu enrollment/test FSDD cũ và hai nguồn ICA ngắn đã được xóa. Dữ liệu ICA mới nằm ở `separation/ami23s/`: hai array mic thật + hai headset tham chiếu đồng bộ 23,28s, cùng hai nguồn solo 3s từ hội thoại AMI. Xem README trong thư mục đó.
