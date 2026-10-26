# Đặc tả Kỹ thuật & Kế hoạch Triển khai (SPEC) — CSE457 Lab 3

Tài liệu này định nghĩa kiến trúc, tham số hệ thống, quy trình thực nghiệm và danh sách đầu việc cho dự án **Nhận dạng Tiếng nói Nâng cao với HMM**. Trạng thái các đầu việc ở Mục 7 phản ánh đúng mã nguồn hiện có.

---

## 1. Mục tiêu Hệ thống

Xây dựng pipeline nhận dạng $5$ từ lệnh tiếng Việt tách rời (`mo`, `dong`, `len`, `xuong`, `dung`) bằng **Whole-word Left-to-Right Gaussian HMM**, phân tích đường trạng thái **Viterbi**, đối sánh công bằng với **DTW Baseline (Lab 2)** và mở rộng giải mã kết hợp **Lexicon + Bigram Language Model**.

## 2. Kiến trúc & Nguyên tắc Thiết kế

```text
src/            Thư viện lõi (không có I/O theo kiểu "chạy là in"): mọi logic nằm ở đây
scripts/        Mỗi file là một bước pipeline, chỉ làm: parse tham số -> gọi src -> ghi results/
run_lab03.sh    Điểm vào production: dựng .venv, chạy các bước theo thứ tự, ghi log
Lab03.ipynb     Chạy từng script bằng main([...]) rồi trực quan hóa/phân tích; không chứa logic xử lý
tests/          pytest trên dữ liệu tổng hợp (không phụ thuộc dataset)
```

Quy tắc:
* `scripts/*` và `Lab03.ipynb` chỉ **import từ `src`**; `src` không import ngược từ `scripts`.
* Mọi script chạy từ thư mục gốc bằng `python -m scripts.<tên>` và có hàm `main(argv=None) -> dict` để notebook gọi trực tiếp.
* Cấu hình dùng chung nằm trong `src/config.py` (`FeatureConfig`, `HMMConfig`, `LMConfig` là dataclass `frozen`). Bước `train` ghi cấu hình vào `models/meta.json`; các bước sau **đọc lại từ đó** để đảm bảo đặc trưng lúc đánh giá khớp lúc huấn luyện.
* Kết quả tái lập được: `random_state` cố định, nhiễu E4/E5/E6 dùng seed cố định, `PYTHONHASHSEED=0` trong script.

## 3. Đặc tả Module `src/`

| Module | Trách nhiệm | API chính |
| :--- | :--- | :--- |
| `config.py` | Đường dẫn, nhãn, siêu tham số | `LABELS`, `DISPLAY`, `FeatureConfig`, `HMMConfig`, `LMConfig` |
| `data.py` | Liệt kê và kiểm tra WAV | `list_split`, `load_splits`, `validate_dataset` |
| `features.py` | Tiền xử lý + đặc trưng | `extract_feat`, `features_from_signal`, `trim_bounds`, `build_featset`, `by_label` |
| `noise.py` | Nhiễu trắng theo SNR | `add_white_noise` |
| `acoustic.py` | Khởi tạo/huấn luyện/kiểm tra/lưu HMM | `make_lr_hmm`, `fit_word_model`, `train_models`, `check_topology`, `save_models`, `load_models`, `load_meta` |
| `decode.py` | Forward score, Viterbi | `score_all`, `recognize`, `viterbi_path`, `state_durations` |
| `dtw.py` | Baseline DTW | `dtw_distance`, `DTWRecognizer` |
| `lm.py` | Lexicon, bigram, re-rank | `parse_lexicon`, `read_corpus`, `BigramLM`, `rerank` |
| `rejection.py` | Ngưỡng từ chối (train-only) | `loo_scores`, `youden_threshold` |
| `metrics.py` | Accuracy, confusion, cặp nhầm lẫn | `accuracy`, `confusion`, `most_confused` |
| `experiments.py` | Thí nghiệm E1-E6 | `e1_states` ... `e6_lm` (trả về `DataFrame`) |
| `plots.py`, `reporting.py` | Hình và bảng Markdown | `plot_*`, `build_figures`, `build_summary` |

### 3.1. Dữ liệu (`dataset/`)
* **Từ vựng ($V = 5$):** `mo` (mở), `dong` (đóng), `len` (lên), `xuong` (xuống), `dung` (dừng).
* **Định dạng WAV:** mono, $F_s = 16000\text{ Hz}$, $16\text{-bit PCM}$ (được `check_data` kiểm tra; sai định dạng là lỗi).
* **Phân chia:** `dataset/train/<label>/` và `dataset/test/<label>/`, tên file `<label>_NN.wav`. Khuyến nghị $6$ train + $\ge 2$ test mỗi lớp. Hiện có $6$ train + $1$ test mỗi lớp (một file `_06` mỗi lớp được chuyển từ test sang train để đủ $6$ mẫu train, đổi lại test chỉ còn $5$ utterance); thiếu so với khuyến nghị chỉ gây **cảnh báo**, mã nguồn không giả định số file cố định.
* **Tập mismatch (tùy chọn):** `dataset/test_mismatch/<label>/*.wav` (người nói khác, khoảng cách mic khác). Nếu có file, E5 tự động thêm điều kiện `test_mismatch`. Nhiễu trắng được tạo tại chỗ nên không cần lưu file.
* **Chống rò rỉ:** `check_data` báo lỗi nếu file test/mismatch trùng nội dung (SHA-1) với file train. Mọi siêu tham số và ngưỡng từ chối chỉ được chọn từ train.

### 3.2. Tiền xử lý & đặc trưng (`features.py`)
* Cắt lặng: `librosa.effects.trim(top_db=30)`; giữ nguyên tín hiệu nếu phần còn lại ngắn hơn một frame.
* Pre-emphasis: $y'[n] = y[n] - 0.97\,y[n-1]$.
* STFT/Mel: cửa sổ Hamming $25\text{ ms}$ ($400$ mẫu), hop $10\text{ ms}$ ($160$ mẫu), `n_fft=512`, `n_mels=26`, $20\text{--}8000\text{ Hz}$.
* Vector $39$ chiều/frame: $13$ MFCC (gồm $c_0$) + $\Delta$ + $\Delta^2$ (`width=9`, `mode="nearest"` để an toàn với utterance ngắn). `use_delta=False` cho $13$ chiều.
* CMVN theo utterance: $\hat{F}_{t,d} = (F_{t,d} - \mu_d) / (\sigma_d + 10^{-8})$.
* Đặc trưng sạch được cache theo `(đường dẫn, FeatureConfig)`; mảng cache ở chế độ chỉ đọc.

### 3.3. Gaussian HMM (`acoustic.py`)
* **Topology:** $N = 5$ (khảo sát $\{3,5,7\}$ ở E1); $\pi = [1,0,\dots,0]$; $a_{i,i} = 0.7$, $a_{i,i+1} = 0.3$ với $i < N-1$; $a_{N-1,N-1} = 1$.
* **Emission:** một Gaussian mỗi trạng thái, `covariance_type="diag"`, `min_covar=1e-3`.
* **Khởi tạo mean/cov:** mặc định `init="uniform"` — chia đều mỗi utterance thành $N$ đoạn theo thời gian rồi lấy thống kê từng đoạn (flat start), phù hợp với topology trái-sang-phải. `init="kmeans"` quay về `init_params="mc"` của hmmlearn; K-means không gắn trạng thái với thứ tự thời gian. Đo trên dữ liệu hiện có: `uniform` đạt accuracy test $1.0$ và không phụ thuộc seed, còn `kmeans` đạt $0.9$ ở `random_state=0` và suy biến ở `random_state=1` (một trạng thái không nhận frame nào -> `transmat_` chứa NaN, `fit_word_model` báo `RuntimeError`). *Đây là điểm khác so với bản SPEC cũ chỉ định `init_params="mc"`.*
* **Baum-Welch:** `n_iter=80`, `tol=1e-3`, `params="tmc"` ($\pi$ cố định), `random_state=0`; ghép các utterance cùng lớp thành `X_concat` + `lengths`.
* **Kiểm tra hậu huấn luyện (`check_topology`):** hội tụ, `transmat_` không có phần tử khác $0$ ở dưới đường chéo (đi lùi) hoặc cách đường chéo $\ge 2$ (nhảy cóc), các hàng tổng bằng $1$. Vi phạm topology làm bước `train` dừng với lỗi; chưa hội tụ chỉ cảnh báo.
* Lưu: `models/hmm_<label>.pkl` + `models/meta.json`. (Pickle chỉ nạp file do pipeline này sinh ra.)

### 3.4. Giải mã & Đánh giá (`decode.py`, `evaluate_hmm`)
* Điểm Forward $s_w(X) = \log P(X|\lambda_w)$ bằng `model.score`; $\hat{w} = \arg\max_w s_w(X)$; điểm theo frame $\tilde{s}_w = s_w / T$.
* Margin $= s_{(1)} - s_{(2)}$.
* Viterbi bằng `model.decode(F, algorithm="viterbi")`, thực hiện trên mô hình của **từ đúng** để quan sát gióng hàng; thống kê số frame dừng ở mỗi trạng thái.
* **Ngưỡng từ chối:** điểm theo frame của leave-one-out trên tập train (mô hình đúng lớp được huấn luyện lại không có utterance đang chấm), chọn ngưỡng tối đa hóa $\text{TPR} - \text{FPR}$ (Youden). Báo cáo tỉ lệ test được chấp nhận. Chưa có dữ liệu OOV để đo tỉ lệ từ chối đúng thực tế.
* Thời gian suy luận đo bằng `time.perf_counter` trên đặc trưng đã tính sẵn (không gồm trích xuất đặc trưng), có một lần chạy khởi động.

### 3.5. Baseline DTW (`dtw.py`, `evaluate_dtw`)
* Cùng đặc trưng $39$-D, cùng tập train (mọi file train là template) và cùng tập test với HMM.
* Khoảng cách Euclid giữa frame; bước đi đối xứng $(i{-}1,j), (i,j{-}1), (i{-}1,j{-}1)$; chuẩn hóa theo $T_a + T_b$; nhận dạng 1-NN (khoảng cách nhỏ nhất tới template của từng lớp).
* Cài đặt vector hóa theo hàng (`cumsum` + `minimum.accumulate`), được test đối chiếu với bản vòng lặp chuẩn.
* "Thời gian train" của DTW là thời gian lưu template (không có pha học); của HMM là tổng thời gian EM. Cả hai không gồm trích xuất đặc trưng.

### 3.6. Lexicon & Bigram LM (`lm.py`, `lm_rerank`)
* `lexicon.txt`: `WORD PHONE_1 ... PHONE_K`, `#` là chú thích; parser báo lỗi dòng thiếu âm vị hoặc từ khai báo trùng; script kiểm tra đủ $5$ từ vựng.
* `corpus.txt`: mỗi dòng một câu lệnh; thêm ký hiệu `<s>` đầu câu.
* Bigram Add-$k$ ($k = 0.5$): $P(w_c | w_p) = \dfrac{C(w_p, w_c) + k}{C(w_p) + kV}$, trong đó $C(w_p)$ là số lần $w_p$ **đứng làm lịch sử** (kể cả khi là từ cuối câu) để phân phối chuẩn hóa về $1$.
* Re-rank MAP: $\text{Score}(w) = \tfrac{1}{T}\log P_{\text{AM}}(X|w) + \lambda_{\text{LM}} \log P_{\text{LM}}(w|w_{\text{prev}})$, $\lambda_{\text{LM}} \in \{0, 0.1, 0.3, 1.0\}$.

## 4. Giao thức Thí nghiệm

| Mã | Biến khảo sát | Cấu hình | Giao thức |
| :-: | :--- | :--- | :--- |
| **E1** | `n_states` | $\{3,5,7\}$ | Accuracy train/test, số mô hình hội tụ, số vòng EM, LL/frame train & test |
| **E2** | Chiều đặc trưng | $13$-D vs $39$-D | Như E1 (LL của hai cấu hình không so sánh trực tiếp được vì khác số chiều) |
| **E3** | Covariance | `diag` vs `full` | Thêm `ll_gap = train_ll − test_ll` để đo overfit |
| **E4** | Số mẫu train/lớp | $\{1,2,3,4,6\}$ (bị chặn bởi số mẫu có) | HMM vs DTW; mỗi mức lấy ngẫu nhiên có seed, lặp 5 lần, báo trung bình ± độ lệch chuẩn |
| **E5** | Mismatch | clean, SNR $\{20,10,5,0\}$ dB, `test_mismatch` (nếu có) | Mô hình huấn luyện trên dữ liệu sạch; nhiễu trắng cộng vào test, lặp 3 seed. SNR tính trên phần tiếng nói (các mẫu trong $30\text{ dB}$ dưới đỉnh) |
| **E6** | $\lambda_{\text{LM}}$ | $\{0,0.1,0.3,1.0\}$ | Mỗi utterance thử với mọi từ đứng trước nó trong corpus (kể cả `<s>`); chạy ở clean và SNR $\{10,5,0\}$ dB |

Lưu ý khi diễn giải: tập test chỉ có $5$ utterance nên một lỗi đổi accuracy $20\%$ và khác biệt nhỏ **không có ý nghĩa thống kê**. Khi nhiễu, bước `trim(top_db=30)` cố định không còn tách được tiếng nói khỏi nền, khiến cả HMM (không có mô hình im lặng) lẫn DTW suy giảm mạnh; đây là hạn chế của front-end cần nêu trong phân tích E5/E6.

## 5. Giao diện Dòng lệnh

```bash
./run_lab03.sh                  # check train eval dtw lm exp report
./run_lab03.sh train eval       # chạy một số bước
./run_lab03.sh test | notebook | clean
python -m scripts.train --n-states 7 --cov-type diag
python -m scripts.experiments --only E1 E3
```

| Bước | Script | Đầu ra chính |
| :--- | :--- | :--- |
| `check` | `scripts.check_data` | `results/data_report.csv` |
| `train` | `scripts.train` | `models/hmm_*.pkl`, `models/meta.json` |
| `eval` | `scripts.evaluate_hmm` | `scores_test.csv`, `confusion_matrix_hmm.csv`, `viterbi_paths.json`, `reject_scores.csv`, `metrics_hmm.json` |
| `dtw` | `scripts.evaluate_dtw` | `scores_dtw.csv`, `confusion_matrix_dtw.csv`, `metrics_dtw.json` |
| `lm` | `scripts.lm_rerank` | `bigram_table.csv`, `e6_lm.csv` |
| `exp` | `scripts.experiments` | `e1_states.csv` ... `e5_robustness.csv` |
| `report` | `scripts.report` | `results/*.png`, `results/summary.md` |

## 6. Yêu cầu Phi chức năng
* **Tái lập:** `./run_lab03.sh` chạy từ môi trường trống đến kết quả đầy đủ, mã thoát khác $0$ nếu bất kỳ bước nào lỗi (`set -euo pipefail`), log ở `results/logs/pipeline.log`.
* **An toàn dữ liệu:** không dùng thống kê tập test để chuẩn hóa hay chọn siêu tham số; CMVN tính trong từng utterance.
* **Chất lượng:** `pytest` ($21$ test) bao phủ topology HMM, DTW, bigram, đặc trưng, nhiễu, ngưỡng từ chối.
* Biểu đồ có đầy đủ nhãn trục và chú thích.

---

## 7. Checklist Triển khai

### Giai đoạn 1: Khởi tạo & Dữ liệu
- [x] `requirements.txt`, `.venv` dựng tự động bởi `run_lab03.sh`.
- [x] Cấu trúc thư mục `dataset/{train,test}`, `models/`, `results/`, `src/`, `scripts/`, `tests/`.
- [x] Đủ $6$ file train mỗi lớp.
- [ ] Thu thêm file test (hiện $1$/lớp, khuyến nghị $\ge 2$) và tập `dataset/test_mismatch/` (người nói khác) từ nguồn *không phải train*.
- [x] `lexicon.txt`, `corpus.txt`.

### Giai đoạn 2: Core Pipeline
- [x] **[A]** `extract_feat(path, cfg, use_delta)` hỗ trợ $13$-D/$39$-D; heatmap `results/mfcc_comparison.png`.
- [x] **[B]** `make_lr_hmm(...)` với $\pi$, $A$ trái-sang-phải.
- [x] **[C]** Huấn luyện $5$ mô hình, kiểm tra hội tụ/topology, in `transmat_`, lưu `.pkl`.
- [x] **[D]** `recognize`, bảng log-likelihood + margin, đồ thị Viterbi `results/viterbi_paths.png`.

### Giai đoạn 3: Đánh giá & Mở rộng
- [x] **[E]** DTW baseline, bảng đối chiếu thời gian/accuracy.
- [x] **[F]** `parse_lexicon`.
- [x] **[G]** `BigramLM`, `rerank`.
- [x] **[H]** Confusion matrix, phân bố điểm và ngưỡng từ chối (train-only).

### Giai đoạn 4: Thí nghiệm
- [x] E1, E2, E3, E4, E5, E6 (số liệu ở `results/summary.md`).
- [x] E4/E5 đã chạy lại với $6$ file train.
- [ ] Chạy lại E4/E5 khi có thêm file test và tập `test_mismatch`.

### Giai đoạn 5: Đóng gói
- [x] README cập nhật số liệu và hình từ `results/`.
- [x] `Lab03.ipynb` chạy từng bước và trực quan hóa.
- [x] `REPORT.md`: phân tích kết quả E1–E6, lỗi và hạn chế.
- [ ] Trả lời $10$ câu hỏi lý thuyết ở Mục 6.1 của đề bài (đề bài chưa có trong repo).

---

## 8. Tiêu chí Nghiệm thu (Rubric 10.0 điểm)

| Hạng mục | Trọng số | Điều kiện đạt chuẩn |
| :--- | :---: | :--- |
| **1. Dữ liệu & Đặc trưng $39$-D** | $1.5$ | Ma trận $T \times 39$, có trim, CMVN đúng, không rò rỉ train/test. |
| **2. Cấu hình & Topology HMM** | $2.0$ | Đúng $\pi$, $A$ trái-sang-phải, Gaussian `diag`, `transmat_` sau `fit` không có bước nhảy lùi. |
| **3. Huấn luyện, Likelihood & Viterbi** | $2.0$ | Pipeline chạy trơn tru, có bảng log-likelihood và biểu đồ gióng hàng Viterbi hợp lý. |
| **4. Đánh giá & So sánh DTW** | $1.5$ | Accuracy, Confusion Matrix, đối chiếu công bằng với DTW trên cùng tập dữ liệu. |
| **5. Lexicon & Bigram LM** | $1.0$ | `lexicon.txt` đúng chuẩn, parser hoạt động, kết quả re-ranking theo $\lambda_{\text{LM}}$. |
| **6. Thí nghiệm E1–E6 & Phân tích lỗi** | $1.0$ | Đủ số liệu, giải thích nhầm lẫn dựa trên số liệu thực. |
| **7. Mã nguồn, Báo cáo & Tái lập** | $1.0$ | Code sạch, chạy lại từ đầu đến cuối không lỗi, biểu đồ đủ nhãn trục. |
