> **Trạng thái kiểm tra:** Phần HMM và Neural TTS bên dưới chỉ là mô phỏng; không có HMM được huấn luyện hay mô hình neural pretrained. Các mô tả, dung lượng model và kết luận hiệu năng cũ tương ứng mô phỏng và không chứng minh yêu cầu pretrained. Xem [REPORT.md](REPORT.md) để biết trạng thái và số đo hiện tại. Điểm nghe chưa được thu thập.

# Lộ trình Triển khai Dự án: Lab 4 - Text-to-Speech (TTS)

Tài liệu theo dõi tiến độ tổng thể và danh mục nhiệm vụ chi tiết của bài thực hành **Lab 4: Tổng hợp tiếng nói (Học phần CSE457 - Xử lý âm thanh và tiếng nói, Trường Đại học Thủy Lợi)**.

---

## 📊 Tổng quan Tiến độ 7 Giai đoạn

- [x] **Giai đoạn 1**: Chuẩn bị Dữ liệu & Chuẩn hóa Văn bản (Front-End Normalization & Inventory)
- [x] **Giai đoạn 2**: Tổng hợp Ghép nối Cơ sở & Làm mượt Biên nối (Baseline Concatenation & Cross-fade)
- [x] **Giai đoạn 3**: Tối ưu hóa Unit Selection bằng Quy hoạch động (Viterbi Search)
- [ ] **Giai đoạn 4**: Nghiên cứu Ngữ điệu & Mô hình Nguồn – Lọc (Prosody, F0 & Source–Filter)
- [ ] **Giai đoạn 5**: Khảo sát Tổng hợp Tham số HMM (HMM-based Generation Mode)
- [ ] **Giai đoạn 6**: Suy luận TTS Học sâu Offline (Pretrained Neural TTS Inference)
- [ ] **Giai đoạn 7**: Đánh giá Tổng hợp, Thí nghiệm Đối sánh & Báo cáo Kỹ thuật

---


## 🎯 Chi tiết Từng Giai đoạn & Nhiệm vụ

### Giai đoạn 1: Chuẩn bị Dữ liệu & Tiền xử lý Văn bản
- [x] **1.1. Thiết lập môi trường**: Khởi tạo `.venv`, cài đặt `requirements.txt`, kiểm tra thư viện âm thanh (`soundfile`, `librosa`, `scipy`).
- [x] **1.2. Chuẩn hóa văn bản (`src/text_norm.py`)**:
  - Viết hàm `normalize_queue_text(text: str) -> list[str]`.
  - Tách token chữ số, xử lý quy tắc đọc số miền xếp hàng (ví dụ: `"23"` $\to$ `["hai", "ba"]`).
  - Viết unit tests kiểm thử bộ chuẩn hóa văn bản (`tests/test_text_norm.py` - 100% PASS).
- [x] **1.3. Xây dựng kho đơn vị âm thanh (`data/units/`)**:
  - Đã sắp xếp 58 file WAV của 16 từ vựng về cấu trúc phẳng chuẩn `data/units/<word>_<id>.wav`.
  - Đã chạy chuẩn hóa toàn bộ file về Mono, 16 kHz, 16-bit PCM, cắt gọn silence và cân bằng RMS = 0.08.
- [x] **1.4. Trích xuất Metadata & Biểu đồ thống kê**:
  - Đã xuất bảng `data/metadata.csv` (58 units đầy đủ thông số duration, rms, sr).
  - Đã xuất biểu đồ **Hình 01**: `figures/01_inventory_stats.png`.

---

### Giai đoạn 2: Baseline Ghép nối & Làm mượt Biên nối
- [x] **2.1. Baseline Concatenation (`src/concatenative.py`)**:
  - Chọn cố định candidate `_01` cho từng token, ghép nối trực tiếp không cross-fade (`xfade_ms = 0`).
  - Đã xuất file âm thanh: `audio_output/raw_concat.wav`.
  - Bộ kiểm thử `tests/test_concatenative.py` đạt 100% PASS.
- [x] **2.2. Phân tích gián đoạn tại biên nối**:
  - Vẽ waveform và spectrogram đánh dấu ranh giới ghép nối từ qua hàm `plot_boundary_spec()`.
  - Phân tích các hiện tượng méo: tiếng click, energy mismatch, pitch jump, spectral discontinuity.
- [x] **2.3. Cài đặt thuật toán Cross-fade (`src/smoothing.py`)**:
  - Cài đặt hàm trộn tuyến tính `crossfade(a, b, n)` và đo độ gián đoạn phổ $D_{\text{spec}}$.
  - Cài đặt và thực thi **Thí nghiệm 1** (`evaluate_crossfade_experiment` trên 4 mức `0, 5, 15, 30 ms`).
  - Đã xuất file âm thanh: `audio_output/best_concat.wav` (xfade = 15 ms).
  - Đã xuất biểu đồ **Hình 02**: `figures/02_boundary_spec.png` (Waveform và Spectrogram trước/sau cross-fade).
  - Đóng gói kịch bản đo kiểm: `bash scripts/benchmark.sh`.

---

### Giai đoạn 3: Unit Selection & Tối ưu hóa Chi phí
- [x] **3.1. Thiết kế các hàm chi phí (`src/unit_selection.py`)**:
  - Target Cost ($C_{\text{target}}$): Độ lệch duration & RMS so với mục tiêu.
  - Join Cost ($C_{\text{join}}$): Độ lệch phổ $D_{\text{spec}}$ trên cửa sổ Hanning 25 ms kết hợp RMS jump.
- [x] **3.2. Thuật toán tìm kiếm chuỗi tối ưu**:
  - Cài đặt thuật toán Dynamic Programming (Viterbi) tối thiểu hóa $\sum C_{\text{target}} + \lambda \sum C_{\text{join}}$.
  - Bộ kiểm thử `tests/test_unit_selection.py` và `tests/test_run_phase3.py` đạt 100% PASS.
- [x] **3.3. Thực nghiệm Unit Selection (Thí nghiệm 2)**:
  - So sánh chi phí và chất lượng giữa Fixed `_01`, Fixed `_02` và Viterbi Optimal.
  - Viterbi giảm tổng chi phí **-40.9%** so với baseline cố định.
  - Đã xuất biểu đồ **Hình 03**: `figures/03_unit_selection.png` (Grouped Bar Chart chi phí & Sơ đồ lưới Trellis).
  - Đã cập nhật file âm thanh tối ưu toàn cục: `audio_output/best_concat.wav` (chuỗi Viterbi + 15ms cross-fade).


---

### Giai đoạn 4: Ngữ điệu, Mô hình Nguồn – Lọc & Voice Conversion
- [x] **4.1. Kiểm chứng tần số cơ bản $F_0$ (`src/prosody.py`)**:
  - Trích đoạn tín hiệu hữu thanh (voiced) của từ `ba_01.wav`, tìm chu kỳ pitch $N_p = 93$ mẫu qua Autocorrelation $R_{xx}[k]$.
  - Kiểm chứng công thức $F_0 \approx F_s / N_p$: Kết quả $F_0 = 172.04\text{ Hz}$ so với YIN $171.34\text{ Hz}$ (sai số chỉ **0.41%**).
- [x] **4.2. Thí nghiệm 3 (Co giãn thời lượng)**:
  - Co giãn tốc độ nói bằng Phase Vocoder ở các mức: `0.8x` (19.24s) và `1.2x` (12.82s).
  - Giữ nguyên cao độ và formant, xuất ra `audio_output/speed_0.8x.wav` và `audio_output/speed_1.2x.wav`.
- [x] **4.3. Thí nghiệm 4 (Dịch cao độ semitone)**:
  - Dịch pitch ở các mức `-2.0` semitone (trầm) và `+2.0` semitone (bổng) theo công thức $r = 2^{\Delta s / 12}$.
  - Giữ nguyên thời lượng, xuất ra `audio_output/pitch_minus2.wav` và `audio_output/pitch_plus2.wav`.
- [x] **4.4. Mở rộng Voice Conversion (Cách 1: Xử lý Tín hiệu & Prosody)**:
  - Giả lập giọng Nữ: Nâng $F_0$ $+5.5$ semitones kết hợp High-shelf Formant Warping -> `audio_output/voice_female_sim.wav`.
  - Mô phỏng giọng Miền Nam: Kéo dài nguyên âm âm tiết cuối $+25\%$ và làm mềm đường nét ngữ điệu -> `audio_output/voice_southern_sim.wav`.
  - Đã xuất biểu đồ **Hình 04**: `figures/04_prosody_analysis.png` (Autocorrelation, so sánh phổ tốc độ/cao độ, và phổ Voice Conversion).
  - Đóng gói kịch bản: `python3 -m src.run_phase4`, bộ kiểm thử `tests/test_prosody.py` & `tests/test_run_phase4.py` đạt 100% PASS.

---

### Giai đoạn 5: Tổng hợp Tham số HMM (HMM-based Generation)
- [x] **5.1. Tích hợp backend HMM (`src/hmm_tts.py`)**:
  - Vận hành HMM ở chế độ sinh (generation mode) với tham số tĩnh $c_t$ (F0, F1, F2, F3, Energy) và tham số động $\Delta c_t$.
  - Thuật toán MLPG giải hệ phương trình $(I + \lambda D^T D) c = \mu$ làm trơn triệt để bước nhảy trạng thái (giảm bước nhảy đột ngột F2 tới **51.2%**).
  - Xuất wrapper `hmm_tts_backend.py` ở root tương thích chuẩn SPEC Section 4.5.
- [x] **5.2. Trích xuất & Vẽ quỹ đạo âm học**:
  - Đã xuất biểu đồ **Hình 04**: `figures/04_hmm_trajectories.png` (Đường cong $F_0$ contour, formant tracks $F_1, F_2$ gắn nhãn từ và spectrogram).
- [x] **5.3. Điều khiển tham số & Xuất âm thanh**:
  - Thử nghiệm can thiệp tốc độ 1.2x (`audio_output/hmm_tts_speed1.2x.wav`) và dịch pitch 1.1x (`audio_output/hmm_tts_pitch1.1x.wav`).
  - Xuất file âm thanh tổng hợp: `audio_output/hmm_tts.wav` (Latency: **54.25 ms**, Dung lượng model ~250 KB).
  - Bộ kiểm thử `tests/test_hmm_tts.py` & `tests/test_run_phase5.py` đạt 100% PASS.

---

### Giai đoạn 6: TTS Học sâu Offline (Pretrained Neural TTS)
- [x] **6.1. Tích hợp wrapper Neural TTS (`src/neural_tts.py`)**:
  - Kiến trúc Acoustic Model 80-band Mel-spectrogram + Fast Neural Vocoder tái tạo pha qua Griffin-Lim momentum.
  - Xuất wrapper `neural_tts_backend.py` ở root tương thích chuẩn SPEC Section 4.6.
- [x] **6.2. Suy luận trên 3 câu kiểm thử**:
  - Xuất file âm thanh: `audio_output/neural_tts.wav` (Câu 1).
  - Trích xuất biểu đồ Mel-spectrogram trung gian: `figures/05_neural_melspec.png`.
- [x] **6.3. Đo đạc hiệu năng**:
  - Đo chính xác thời gian trễ trên 3 câu: Câu 1 (1939.1 ms, RTF 0.4013x), Câu 2 (713.4 ms, RTF 0.1476x), Câu 3 (814.6 ms, RTF 0.1907x).
  - Latency trung bình: **1155.72 ms**, RTF trung bình: **0.2465x** (Nhanh hơn thời gian thực gấp ~4 lần).
  - Bộ kiểm thử `tests/test_neural_tts.py` & `tests/test_run_phase6.py` đạt 100% PASS.

---

### Giai đoạn 7: Đánh giá Tổng hợp, Thí nghiệm 5 & Báo cáo Kỹ thuật
- [ ] **7.1. Thí nghiệm 5 (Đối kháng 3 phương pháp)**:
  - Chuẩn hóa âm lượng đồng đều trên 3 câu kiểm thử.
  - Thực hiện listening test mù (5–10 người nghe), đánh giá *Intelligibility (%)*, *Naturalness (MOS 1–5)*, *Continuity*.
  - Lưu kết quả đánh giá vào `ratings.csv`.
- [ ] **7.2. Biểu đồ so sánh tổng hợp**:
  - Điền đầy đủ bảng tổng hợp 3 phương pháp (Độ rõ, Độ tự nhiên, Độ liên tục, Độ trễ, Dung lượng lưu trữ).
  - Xuất biểu đồ **Hình 05**: `figures/05_methods_comparison.png` (Biểu đồ radar / bar chart).
- [ ] **7.3. Hoàn thiện kiểm thử tự động**:
  - Kiểm tra chạy thông suốt `bash scripts/run_pipeline.sh` và `bash scripts/benchmark.sh`.
  - Kiểm tra tương tác đầy đủ trên `notebooks/Lab04.ipynb`.
- [ ] **7.4. Soạn thảo Báo cáo Kỹ thuật (`report.pdf`, 5–8 trang)**:
  - Trình bày cơ sở lý thuyết, công thức và thiết kế thí nghiệm.
  - Tổng hợp số liệu thực nghiệm từ 5 thí nghiệm bắt buộc kèm phân tích biểu đồ.
  - Trả lời đầy đủ, chính xác 10 câu hỏi lý thuyết & chuyên môn trong Mục 5.2 của `SPEC.md`.

---

## 📦 Bảng Kiểm Nghiệm thu Sản phẩm Nộp (Deliverables Checklist)

| STT | Tên sản phẩm | Vị trí lưu trữ | Trạng thái |
| :-: | :--- | :--- | :-: |
| 1 | Mã nguồn module & Pipeline | `src/*.py`, `scripts/*.sh` | ⏳ Đang triển khai |
| 2 | Sổ tay tương tác | `notebooks/Lab04.ipynb` | ⏳ Đang triển khai |
| 3 | Audio Baseline ghép nối | `audio_output/raw_concat.wav` | ⏳ Chưa có |
| 4 | Audio Tối ưu Unit Selection | `audio_output/best_concat.wav` | ⏳ Chưa có |
| 5 | Audio HMM Synthesis | `audio_output/hmm_tts.wav` | ⏳ Chưa có |
| 6 | Audio Pretrained Neural TTS | `audio_output/neural_tts.wav` | ⏳ Chưa có |
| 7 | Biểu đồ Thống kê Inventory | `figures/01_inventory_stats.png` | ⏳ Chưa có |
| 8 | Biểu đồ Biên nối Cross-fade | `figures/02_boundary_spec.png` | ⏳ Chưa có |
| 9 | Biểu đồ Chi phí Unit Selection | `figures/03_unit_selection.png` | ⏳ Chưa có |
| 10 | Biểu đồ Quỹ đạo HMM | `figures/04_hmm_trajectories.png` | ⏳ Chưa có |
| 11 | Biểu đồ So sánh 3 phương pháp | `figures/05_methods_comparison.png` | ⏳ Chưa có |
| 12 | Dữ liệu Chấm điểm Nghe mù | `ratings.csv` | ⏳ Chưa có |
| 13 | Báo cáo Thực hành Kỹ thuật | `report.pdf` (5–8 trang) | ⏳ Chưa có |
