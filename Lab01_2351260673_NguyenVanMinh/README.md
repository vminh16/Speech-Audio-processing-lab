# CSE457 - Xử lý âm thanh và tiếng nói
## Lab 1: Phân tích và xử lý tín hiệu âm thanh số

* **Họ và tên:** Nguyễn Văn Minh
* **Mã số sinh viên (MSSV):** 2351260673
* **Cơ sở đào tạo:** Trường Đại học Thủy lợi

---

### Cấu trúc thư mục dự án

```text
Lab01_2351260673_NguyenVanMinh/
├── .venv/                              # Môi trường ảo Python
├── audio/                              # Dữ liệu âm thanh
│   ├── input_piano.mp3                 # Tệp vào: âm nhạc Piano
│   ├── input_speech.mp3                # Tệp vào: tiếng nói bản tin
│   ├── filtered_*.wav                  # Sau lọc LPF 1k / HPF 1k / BPF 300-3400 (Khối F)
│   ├── quantized_*.wav                 # Sau lượng tử hóa 4 / 8 / 16 bit (Khối G)
│   └── resampled_*.wav                 # Sau resampling 16 kHz / 8 kHz (Khối G)
├── figures/                            # Đồ thị thực nghiệm
│   ├── waveform_{speech,piano}.png     # Dạng sóng (Khối B)
│   ├── fft_{speech,piano}.png          # Phổ FFT (Khối C)
│   ├── spectrogram_{speech,piano}.png  # Spectrogram, khung 10/25/50 ms (Khối D)
│   ├── window.png                      # Rectangular vs Hamming (Khối E)
│   ├── filter_response.png             # |H(f)| và độ trễ nhóm (Khối F)
│   ├── filter_spectrum.png             # Phổ trước/sau lọc (Khối F)
│   ├── quantization.png                # SNR và phổ nhiễu lượng tử (Khối G)
│   └── resampling.png                  # Phổ sau resampling (Khối G)
├── Lab01_2351260673.ipynb              # Notebook thực hành toàn bộ bài Lab
├── report_Lab01.md                     # Báo cáo thực hành
├── requirements.txt                    # Thư viện Python cần cài
└── README.md                           # Giới thiệu và hướng dẫn dự án
```

---

### Hướng dẫn môi trường và thực thi

1. **Cài đặt:** Python 3.12 và `pip install -r requirements.txt`. `pydub` cần [FFmpeg](https://ffmpeg.org) có trong `PATH` để đọc MP3.
2. **Chạy:** Mở [Lab01_2351260673.ipynb](Lab01_2351260673.ipynb), chọn kernel Python đã cài thư viện trên (ví dụ `.venv`), rồi chạy tuần tự từ đầu đến cuối (Run All). Toàn bộ hình trong `figures/` và file WAV trong `audio/` được tạo lại tự động.

---

### Tiến độ các khối nội dung

- [x] **Khối A: Đọc, kiểm tra dữ liệu và trực quan hóa chuỗi âm thanh ban đầu** (Hoàn thành: Hỗ trợ cả 2 tệp Piano và Giọng nói, chuẩn hóa $[-1.0, 1.0]$, so sánh stereo/mono (RMS, dạng sóng), đo đạc Peak/RMS, bảng so sánh `pandas.DataFrame`, trực quan hóa sơ bộ dạng sóng ban đầu).
- [x] **Khối B: Phân tích miền thời gian** (Hoàn thành: Đo đạc Peak, RMS, Energy, phân tích 2 đoạn tương phản năng lượng cao vs. khoảng lặng chênh lệch 25.6 lần, xuất đồ thị tách biệt `waveform_speech.png` và `waveform_piano.png`).
- [x] **Khối C: Phân tích miền tần số bằng FFT** (Hoàn thành: Cửa sổ Hamming, xác định 5 đỉnh phổ chính cho cả Piano và Tiếng nói, khảo sát $N_{\text{FFT}} = 2048$ vs $65536$, phân biệt khoảng cách bin $\Delta f$ và độ phân giải vật lý, xuất `fft_piano.png` và `fft_speech.png`).
- [x] **Khối D: STFT và Spectrogram** (Hoàn thành: STFT tự viết khớp `scipy`, Parseval; khung 10/25/50 ms cho 2 tệp; kiểm chứng trade-off bằng số đỉnh phân tách và khoảng lặng 206.6 ms).
- [x] **Khối E: Thí nghiệm Cửa sổ (Windowing)** (Hoàn thành: main-lobe 40/80 Hz, side-lobe −13.3/−42.7 dB khớp lý thuyết; nền phổ giảm ~16 dB với Hamming).
- [x] **Khối F: Lọc số** (Hoàn thành: FIR 401 taps LPF/HPF/BPF, τ_g = 4.17 ms, độ lợi đo khớp $|H(f)|$, xuất 6 file `filtered_*.wav`).
- [x] **Khối G: Lượng tử hóa, Resampling và Mã hóa** (Hoàn thành: SNR ≈ 6.02 dB/bit, resampling 16/8 kHz + minh họa aliasing, CR = 7.92).
