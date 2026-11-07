> **Trạng thái kiểm tra:** Phần HMM và Neural TTS bên dưới chỉ là mô phỏng; không có HMM được huấn luyện hay mô hình neural pretrained. Các mô tả, dung lượng model và kết luận hiệu năng cũ tương ứng mô phỏng và không chứng minh yêu cầu pretrained. Xem [REPORT.md](REPORT.md) để biết trạng thái và số đo hiện tại. Điểm nghe chưa được thu thập.

# CSE457 - Lab 4: Tổng hợp tiếng nói (Text-to-Speech)

Tài liệu hướng dẫn tổng quan, cấu trúc thư mục chuẩn production và cách thiết lập môi trường cô lập, thực thi thử nghiệm cho bài thực hành **Lab 4: Tổng hợp tiếng nói — Từ kiến trúc TTS, ghép nối và unit selection đến tổng hợp tham số HMM và TTS học sâu** (Học phần CSE457 - Xử lý âm thanh và tiếng nói, Bộ môn Trí tuệ nhân tạo, Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi).

## 1. Giới thiệu bài Lab

Lab 4 kết nối kiến thức xử lý tín hiệu âm thanh, trích xuất đặc trưng và mô hình xác suất với bài toán phát sinh waveform tiếng nói (Text-to-Speech - TTS). Nội dung thực hành tập trung vào việc xây dựng một hệ thống **Mini TTS miền hẹp** cho bài toán *"Thông báo xếp hàng"* (ví dụ: `"Mời khách hàng số 23 đến quầy 5."`) và so sánh trực diện ba phương pháp tổng hợp tiếng nói:

1. **Tổng hợp ghép nối (Concatenative & Unit Selection TTS)**: Chuẩn hóa văn bản (text normalization), xây dựng kho đơn vị âm thanh (word-unit inventory), lựa chọn candidate tối ưu bằng quy hoạch động dựa trên *target cost* và *join cost*, kết hợp làm mượt biên nối bằng *cross-fade / overlap-add (OLA)*.
2. **Tổng hợp tham số dựa trên HMM (HMM-based Parametric Synthesis)**: Khảo sát cơ chế HMM hoạt động ở chế độ phát sinh (generation mode) để sinh quỹ đạo tham số âm học (F0 contour, duration, spectral/formant parameters) có ràng buộc đặc trưng động ($\Delta c_t$).
3. **Tổng hợp tiếng nói học sâu (Pretrained Neural TTS)**: Chạy suy luận (inference) offline theo chuỗi `text → acoustic representation (mel-spectrogram) → vocoder → waveform` và đo độ trễ (latency/RTF).

### Lịch trình phân bổ thực hành (3 tiết)

| Thời lượng | Nội dung trọng tâm | Kết quả đầu ra |
| :--- | :--- | :--- |
| **Tiết 1** | TTS pipeline, text normalization, xây dựng inventory và baseline ghép nối | Sinh câu nói đầu tiên (`raw_concat.wav`) từ các word-unit |
| **Tiết 2** | Unit selection, boundary smoothing, prosody (F0/duration) và HMM generation | Cải thiện đường nối (`best_concat.wav`); quan sát quỹ đạo tham số HMM |
| **Tiết 3** | Inference TTS học sâu; thực hiện listening test, so sánh và viết báo cáo | Bảng so sánh 3 phương pháp, bộ file audio hoàn chỉnh và báo cáo |

## 2. Cấu trúc thư mục dự án (Production-Ready)

Dự án được tổ chức theo chuẩn kiến trúc module tách biệt rõ giữa **mã nguồn (`src/`)**, **kịch bản tự động hóa (`scripts/`)**, **dữ liệu (`data/`)**, **sổ tay nghiên cứu (`notebooks/`)** và **kết quả đầu ra (`audio_output/`, `figures/`)**, đảm bảo tính tái lập (reproducible), dễ bảo trì và kiểm thử tự động:

```text
Lab04_2351260673_NguyenVanMinh/
├── .gitignore                    # Bỏ qua .venv, bytecode cache, file âm thanh tạm thời
├── README.md                     # Tài liệu hướng dẫn cài đặt, cấu trúc và vận hành
├── SPEC.md                       # Đặc tả kỹ thuật, công thức, metrics, yêu cầu và rubric
├── requirements.txt              # Danh sách thư viện và phiên bản phụ thuộc
├── .venv/                        # Môi trường ảo Python độc lập (tự động tạo)
│
├── data/                         # Thư mục chứa dữ liệu đầu vào và tài nguyên âm học
│   ├── metadata.csv              # Thông tin unit: label, speaker, duration, RMS, file_path
│   ├── units/                    # Inventory WAV (mono, Fs = 16 kHz, 16-bit PCM, đã cắt silence)
│   │   ├── moi_01.wav, moi_02.wav
│   │   ├── khach_01.wav, khach_02.wav
│   │   ├── hang_01.wav, hang_02.wav
│   │   ├── so_01.wav, so_02.wav
│   │   ├── khong_01.wav ... chin_02.wav
│   │   ├── den_01.wav, den_02.wav
│   │   └── quay_01.wav ...
│   └── models/                   # Trọng số offline mô hình HMM / Pretrained Neural TTS
│
├── src/                          # Mã nguồn chính (Source Code đóng gói module)
│   ├── __init__.py               # Khởi tạo package src
│   ├── text_norm.py              # Chuẩn hóa văn bản, tách token và ánh xạ chữ số miền xếp hàng
│   ├── concatenative.py          # Ghép nối đơn vị, tính Target Cost, Join Cost & Unit Selection
│   ├── smoothing.py              # Làm mượt biên nối: Linear Cross-Fade, OLA, Zero-Crossing
│   ├── prosody.py                # Phân tích F0, năng lượng RMS và biến đổi Pitch/Duration (TD-PSOLA)
│   ├── hmm_tts.py                # Backend tổng hợp tham số HMM với ràng buộc delta
│   ├── neural_tts.py             # Wrapper suy luận Neural TTS pretrained chạy offline
│   ├── utils.py                  # Tiện ích đọc/ghi audio, vẽ phổ Spectrogram, tính metrics
│   └── pipeline.py               # Entrypoint điều phối toàn bộ luồng thực nghiệm A -> G
│
├── scripts/                      # Kịch bản thực thi và tự động hóa (.sh)
│   ├── setup_env.sh              # Tự động khởi tạo .venv và cài đặt dependencies
│   ├── run_pipeline.sh           # Chạy toàn bộ pipeline từ đầu đến cuối xuất kết quả
│   └── benchmark.sh              # Đo kiểm thời gian thực thi (Latency), RTF và so sánh metrics
│
├── notebooks/                    # Sổ tay tương tác phục vụ nghiên cứu & trực quan hóa
│   └── Lab04.ipynb               # Jupyter Notebook thực hành tương tác từng bước
│
├── audio_output/                 # Thư mục lưu các file âm thanh kết quả tổng hợp
│   ├── raw_concat.wav            # Audio ghép nối trực tiếp (baseline, xfade = 0 ms)
│   ├── best_concat.wav           # Audio tối ưu hóa với Unit Selection và Cross-Fade
│   ├── hmm_tts.wav               # Audio sinh từ backend HMM synthesis
│   └── neural_tts.wav            # Audio sinh từ Pretrained Neural TTS offline
│
├── figures/                      # Thư mục lưu tối thiểu 5 biểu đồ phân tích chuẩn
│   ├── 01_inventory_stats.png    # Histogram phân bố duration và RMS của inventory
│   ├── 02_boundary_spec.png      # Waveform và Spectrogram biên nối trước/sau cross-fade
│   ├── 03_unit_selection.png     # So sánh Join Cost và Total Cost giữa các chuỗi candidate
│   ├── 04_hmm_trajectories.png   # Quỹ đạo F0 contour và tham số phổ/formant theo frame
│   └── 05_methods_comparison.png # Biểu đồ radar / cột so sánh 3 phương pháp TTS
│
├── ratings.csv                   # Dữ liệu chấm điểm listening test (MOS & Pairwise A/B)
└── report.pdf                    # Báo cáo thực hành chi tiết (5–8 trang)
```

## 3. Thiết lập môi trường với Virtual Environment (.venv)

Để đảm bảo tính độc lập, tránh xung đột thư viện hệ thống và tuân thủ chuẩn phát triển phần mềm, dự án bắt buộc sử dụng **Python Virtual Environment (`.venv`)** với **Python 3.8+** (khuyến nghị Python 3.10 hoặc 3.11).

### Cách 1: Sử dụng script tự động (Khuyến nghị)
Chạy script bash đã chuẩn bị sẵn để tự động tạo `.venv`, kích hoạt và cài đặt gói:
```bash
# Cấp quyền thực thi và chạy script cài đặt
chmod +x scripts/*.sh
bash scripts/setup_env.sh
```

### Cách 2: Thiết lập thủ công từng bước

1. **Khởi tạo môi trường ảo `.venv`**:
   ```bash
   python3 -m venv .venv
   ```

2. **Kích hoạt môi trường ảo**:
   - Trên Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```
   - Trên Windows (PowerShell / Command Prompt):
     ```powershell
     .venv\Scripts\activate
     # Hoặc PowerShell: .venv\Scripts\Activate.ps1
     ```

3. **Nâng cấp pip và cài đặt thư viện phụ thuộc**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Danh mục thư viện chính (`requirements.txt`)
- `numpy`, `scipy`: Xử lý mảng số, đại số tuyến tính, biến đổi tín hiệu và STFT.
- `soundfile`: Đọc và ghi file âm thanh WAV chuẩn 16-bit PCM.
- `librosa`: Trích xuất đặc trưng âm thanh (MFCC, Spectrogram, F0, RMS).
- `matplotlib`: Trực quan hóa dữ liệu, vẽ waveform và spectrogram.
- `tabulate`: Hiển thị bảng chi phí và kết quả benchmark trực quan.
- `jupyterlab` / `notebook`: Môi trường sổ tay tương tác.

> **Lưu ý quan trọng**: Tuyệt đối không tự ý tải các mô hình Neural TTS trực tuyến dung lượng lớn gây quá tải mạng phòng Lab. Sử dụng mô hình offline chuẩn đã tích hợp trong `data/models/` hoặc lightweight vocoder.

## 4. Hướng dẫn vận hành và thực thi

Sau khi đã kích hoạt môi trường ảo (`source .venv/bin/activate`), có thể thực hiện chạy theo các phương thức sau:

### Phương thức 1: Chạy nhanh qua Shell Scripts (Chuẩn Production & Benchmark)

1. **Chạy toàn bộ pipeline kiểm thử từ đầu đến cuối**:
   ```bash
   bash scripts/run_pipeline.sh
   ```
   *Script sẽ tuần tự chuẩn hóa văn bản, ghép nối baseline, tối ưu unit selection, làm mượt biên nối, chạy HMM/Neural inference, xuất âm thanh vào `audio_output/` và lưu 5 hình biểu đồ vào `figures/`.*

2. **Chạy đo kiểm hiệu năng và benchmark**:
   ```bash
   bash scripts/benchmark.sh
   ```
   *Script đo lường độ trễ (Latency), Real-Time Factor (RTF), Spectral Discontinuity ($D_{\text{spec}}$), dung lượng bộ nhớ (Memory Footprint) và in bảng so sánh trực quan giữa 3 phương pháp.*

### Phương thức 2: Thực thi bằng Python Module (`src/`)

Chạy trực tiếp module chính qua cờ `-m` để đảm bảo định vị đúng package:
```bash
# Chạy pipeline tổng thể
python3 -m src.pipeline

# Hoặc chạy kiểm thử một câu cụ thể với tham số tùy chọn:
python3 -m src.pipeline --text "Mời khách hàng số 42 đến quầy 3." --xfade 20
```

### Phương thức 3: Thực hành tương tác trên Jupyter Notebook

Nếu cần quan sát đồ thị trực quan và debug từng bước trong giờ thực hành:
```bash
jupyter lab
# Hoặc: jupyter notebook notebooks/Lab04.ipynb
```

## 5. Sử dụng API Module (`src/`) trong mã nguồn Python

Các mô-đun trong thư mục `src/` được thiết kế độc lập, chuẩn hóa interface và dễ dàng tái sử dụng:

```python
import soundfile as sf
from src.text_norm import normalize_queue_text
from src.concatenative import build_baseline_sequence, select_optimal_units
from src.smoothing import crossfade_concat
from src.hmm_tts import synthesize_hmm
from src.neural_tts import synthesize_neural

# 1. Chuẩn hóa văn bản miền thông báo
text = "Mời khách hàng số 23 đến quầy 5."
tokens = normalize_queue_text(text)
# Kết quả: ['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']

# 2. Tổng hợp ghép nối Baseline (không làm mượt, xfade = 0ms)
baseline_units = build_baseline_sequence(tokens, inventory_dir="data/units")
y_raw, sr = crossfade_concat(baseline_units, sr=16000, xfade_ms=0)
sf.write("audio_output/raw_concat.wav", y_raw, sr)

# 3. Unit Selection tối ưu chi phí & làm mượt biên nối (Cross-fade 15ms)
best_units = select_optimal_units(tokens, inventory_dir="data/units", weight_join=1.0)
y_best, sr = crossfade_concat(best_units, sr=16000, xfade_ms=15)
sf.write("audio_output/best_concat.wav", y_best, sr)

# 4. HMM-based Parametric Synthesis
y_hmm, sr, hmm_tracks = synthesize_hmm(text)
sf.write("audio_output/hmm_tts.wav", y_hmm, sr)

# 5. Pretrained Neural TTS Offline
y_neural, sr, neural_meta = synthesize_neural(text)
sf.write("audio_output/neural_tts.wav", y_neural, sr)
```

## 6. Kiểm tra nghiệm thu sản phẩm (Checklist)

Trước khi đóng gói nộp bài hoặc tạo commit, kiểm tra các tiêu chí sau:

- [ ] Môi trường ảo `.venv` hoạt động ổn định, không thiếu thư viện trong `requirements.txt`.
- [ ] Kịch bản `bash scripts/run_pipeline.sh` và `bash scripts/benchmark.sh` chạy thông suốt mã lỗi 0.
- [ ] Thư mục `audio_output/` chứa đầy đủ 4 file WAV: `raw_concat.wav`, `best_concat.wav`, `hmm_tts.wav`, `neural_tts.wav`.
- [ ] Thư mục `figures/` chứa đầy đủ tối thiểu 5 biểu đồ phân tích (`01_inventory_stats.png` đến `05_methods_comparison.png`).
- [ ] File `ratings.csv` được cập nhật kết quả khảo sát nghe chủ quan (MOS / A-B test).
- [ ] Báo cáo `report.pdf` (5–8 trang) hoàn thành đầy đủ các mục thực nghiệm và giải thích 10 câu hỏi theo `SPEC.md`.

Xem chi tiết công thức toán học, định nghĩa các chỉ số đánh giá (metrics), quy trình thiết kế 5 thí nghiệm bắt buộc và tiêu chí chấm điểm trong file `SPEC.md`.

