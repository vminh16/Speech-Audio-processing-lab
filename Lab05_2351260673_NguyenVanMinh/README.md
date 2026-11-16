# CSE457 — Lab 5

Speech enhancement, speaker recognition và source separation. Thuật toán nằm trong `src`; notebook gọi chúng để chạy thí nghiệm và visualize.

## Chạy toàn bộ

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=/tmp/lab5-mpl .venv/bin/python tools/run_lab.py
```

Hoặc mở **`notebooks/Lab5.ipynb`**, chọn `.venv/bin/python`, Restart Kernel → Run All. Notebook chính thực thi toàn bộ pipeline, không chỉ đọc ảnh/CSV cũ. Cell config tập trung các tham số; mặc định đúng các thí nghiệm của đề. Không cần Internet khi dữ liệu đã có.

Kết quả chính:

- **`report.pdf`**: báo cáo A4 **6 trang**, công thức/tham số, số đo, ba kết luận và trả lời 12 câu hỏi.
- **`report.md`**: nguồn nội dung báo cáo.
- `results/lab_summary.json`: tổng hợp kết quả và config.
- `results/figures/`: các biểu đồ trong notebook/báo cáo.
- Các WAV, score, trial, model và chỉ số theo từng phần ở bên dưới.

Notebook chi tiết:

| Notebook | Nội dung |
|---|---|
| `notebooks/Lab5.ipynb` | Điểm chạy đầy đủ + case study hội thoại 23s + tạo báo cáo |
| `notebooks/Lab5_Part1_Enhancement.ipynb` | Phần 4.1–4.3, FSDD/noise, SS/Wiener |
| `notebooks/Lab5_Part2_Speaker.ipynb` | Phần 4.4–4.5, AMI MFCC/cosine và demo nhận dạng hội thoại |
| `notebooks/Lab5_Part3_Separation.ipynb` | Phần 4.6–4.8, ICA mô phỏng và hai mic thật 23s |

Notebook Part1 trước đây mang tên Lab5.ipynb; nội dung đã được giữ nguyên dưới tên Part1. Lab5.ipynb hiện là notebook tổng.

## Dữ liệu

- `data/clean/`: 7 FSDD clip dùng enhancement, gồm 6 dev và 1 reference/test. Nguồn 8 kHz, ghép chữ số rồi resample 16 kHz; không gọi là câu nói tự nhiên.
- `data/noise/`: mưa và máy hút bụi ESC-50, mỗi file 5s. Không đổi tên thành noise quạt.
- `data/speaker_recognition/ami/`: AMI EN2009d, 3 người × (4 enroll + 2 dev + 2 test), các lượt nói thủ công không chồng thời gian; preview hội thoại 23,28s không trùng các split.
- `data/separation/ami23s/`: hai mic array thật Array1-01/05, hai headset gần người A/B để tham chiếu và hai solo crops 3s lấy từ preview.
- README/manifest trong `data/` ghi nguồn, timestamp, split, giấy phép và SHA256. Không tải cả cuộc họp dài; chỉ lấy các byte WAV cần thiết.

Nếu cần tạo lại dữ liệu (cần Internet, phần FSDD cần ffmpeg):

```bash
python3 tools/prepare_data.py
python3 tools/prepare_ami.py
.venv/bin/python tools/prepare_separation_data.py
```

FSDD CC BY-SA 4.0; ESC-50 theo LICENSE và metadata upstream; AMI CC BY 4.0, ghi công AMI Consortium / University of Edinburgh. Chi tiết và giấy phép lưu ở `data/**/README.md`, `data/raw/esc50/`, `data/speaker_recognition/ami/metadata/`.

## Phương pháp và logic đánh giá

**Enhancement:** mono 16 kHz, Hann 400/hop160/FFT512. Hai hàm `spectral_subtraction(noisy, ...)`, `wiener_filter(noisy, ...)` chỉ nhận observation, không nhận clean/noise gốc/alpha/SNR. Noise PSD ước lượng offline từ 20% frame năng lượng thấp, bù bias ×1,2. SS làm mượt power rồi trừ phổ, chặn gain ≤1; Wiener dùng decision-directed 0,9, amplitude floor 0,05, vẫn H=Ps/(Ps+Pn). Không áp cutoff suy từ ground truth. Chọn β=2, γ=0,01 trên dev. Đây là cải tiến bộ ước lượng của baseline, không phải neural denoiser. Có thể méo/giữ noise nếu giả thiết không phù hợp.

**Speaker:** 20 MFCC gồm c0, pre-emphasis0,97, 40 Mel HTK, log-power/DCT-II, mean/std MFCC và delta9 →40/80 chiều, L2. Model=mean 4 enrollment embeddings rồi L2. MFCC/delta/CMN chọn bằng dev accuracy rồi margin; θ chọn gần FAR=FRR trên dev, khóa trước test. Test gốc 4/6 đúng; FAR2/12, FRR1/6. Cosine gần 1 không phải xác suất. Không dùng test chọn feature/threshold.

**ICA:** `separate_ica(mixtures)` nhận `(samples,2)`, từ chối kênh đồng tuyến như mono nhân đôi. FastICA2, unit-variance whitening, seed0, max_iter2000, tol1e−5. Không nhận A/nguồn gốc/nhãn. Permutation/sign được ghép sau fit để đánh giá. Mô phỏng có ground truth, ma trận tốt/gần suy biến, có/không noise20dB. Thu thật dùng array mics, headset chỉ gần đúng; không báo SNR sạch hay coi hội tụ là tách thành công.

**Tích hợp:** chạy SS/Wiener trên toàn hội thoại 23s rồi cắt cùng 5 lượt đơn để ID; chạy thêm 2 loại noise × 3 SNR trên cả hội thoại và 6 test clip. Model/feature/θ/β/γ đã khóa. SNR chỉ có khi cộng noise có kiểm soát so với bản AMI gốc (vẫn có ambient), không gán clean SNR cho xử lý trực tiếp. Dùng cùng gain cho mọi input/output/reference và đo trên PCM16 đọc lại; gain hiện tại=1. Original/noisy/SS/Wiener có WAV riêng.

Kết quả tích hợp chính: hội thoại gốc ID 3/5→4/5→5/5 (Raw/SS/Wiener), nhưng FRR với θ khóa tăng 40%→60% sau Wiener. Thêm mưa5dB: SNR Wiener≈13,93dB nhưng ID vẫn0/5. Trên 6 test clip, máy hút bụi5dB: ID33,3%→50%. Không kết luận enhancement luôn giúp speaker recognition hay FAR=0 là tốt khi FRR=100%.

Demo có ranh giới thủ công, bỏ overlap/lượt <1s; không phải automatic diarization. Các split cùng phiên và số mẫu nhỏ. Các WAV A/B có sẵn; báo cáo không giả lập điểm nghe chủ quan chưa được đánh giá độc lập.

## Module và kết quả

| Module | Vai trò |
|---|---|
| `src/enhancement.py`, `src/experiments.py` | SS/Wiener, noise, SNR và chọn cấu hình dev |
| `src/speaker.py`, `src/speaker_experiments.py` | MFCC, enrollment, ID/verification, hội thoại |
| `src/separation.py`, `src/separation_experiments.py` | FastICA, mô phỏng và mic thật |
| `src/integration.py` | Enhancement → ID/verification, tham số khóa |
| `src/lab.py` | Pipeline đầy đủ, DEFAULT_CONFIG, figures, summary |
| `src/*visualization.py` | Hàm vẽ, không thay thuật toán |
| `tools/run_lab.py`, `tools/build_report.py` | Điểm chạy CLI và tạo báo cáo |

- `results/audio/enhancement/`:19 WAV, config, bảng SNR và quét β/γ trên dev.
- `results/speaker_recognition/`: model NPZ, test scores/predictions/confusion, FAR–FRR, ablation, timeline hội thoại.
- `results/separation/`:30 WAV, ma trận correlation trước/sau matching, condition/determinant và noise degradation.
- `results/integration/`:154 WAV (gồm 7 references),231 prediction rows,693 claim trials,147 waveform records,42 nhóm so sánh. Có CSV/config của toàn bộ phép đo; không chỉ giữ case tốt.
- `results/figures/integration/`: timeline trước/sau, spectrogram, SNR và ID trên cả hai loại noise.

Tạo lại riêng PDF từ kết quả hiện có: `MPLCONFIGDIR=/tmp/lab5-mpl .venv/bin/python tools/build_report.py`.

Kiểm tra: `.venv/bin/python -m unittest discover -s tests`.

Tham khảo thuật toán: đề Lab5, Huang–Acero–Hon (2001), Rabiner–Schafer (2010); decision-directed Ephraim–Malah (1984): https://malah.net.technion.ac.il/files/2017/08/Ephraim_Speech_Enhancement_ASSP84.pdf ; FastICA: https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.FastICA.html .
