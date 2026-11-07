# CSE457 - Lab 4 Specification: Đặc tả Kỹ thuật & Yêu cầu Thực hành Tổng hợp Tiếng nói (TTS)

Tài liệu đặc tả kỹ thuật chi tiết cho **Lab 4: Tổng hợp tiếng nói (Text-to-Speech)** thuộc học phần **CSE457 - Xử lý âm thanh và tiếng nói** (Bộ môn Trí tuệ nhân tạo, Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi). Tài liệu này quy định mục tiêu đầu ra, nền tảng toán học, hệ thống chỉ số đánh giá (metrics), phương pháp triển khai từng mô-đun, thiết kế thí nghiệm và tiêu chí nghiệm thu.

## 1. Mục tiêu và Chuẩn đầu ra (Learning Outcomes & Objectives)

Mục tiêu cốt lõi của bài Lab là giúp sinh viên kết nối kiến thức xử lý tín hiệu, trích xuất đặc trưng âm học và mô hình xác suất để giải quyết bài toán phát sinh tín hiệu tiếng nói (waveform generation), tự xây dựng một hệ thống **Mini TTS miền hẹp** (*"Thông báo xếp hàng"*) và đánh giá định lượng/định tính chất lượng đầu ra.

### Chuẩn đầu ra cần đạt
1. **Kiến trúc hệ thống**: Mô tả chính xác chuỗi xử lý tổng quát của hệ thống TTS: `Text Analysis → Phonetic Analysis → Prosodic Analysis (F0, Duration, Energy) → Synthesis Backend → Waveform`.
2. **Xử lý văn bản & Ghép nối cơ sở**: Xây dựng bộ chuẩn hóa văn bản (text normalization) cho miền hẹp, ánh xạ chuỗi token sang đơn vị phát âm/ghi âm (word-unit) và phát sinh câu nói bằng phương pháp ghép nối trực tiếp.
3. **Unit Selection**: Giải thích và tính toán chi phí mục tiêu (*target cost*) cùng chi phí ghép nối (*concatenation/join cost*); áp dụng quy hoạch động (dynamic programming) hoặc tìm kiếm vét cạn (exhaustive search) để tìm chuỗi đơn vị âm thanh tối ưu.
4. **Làm mượt biên nối (Boundary Smoothing)**: Khử tiếng nổ/lách cách (click/discontinuity) tại biên ghép nối bằng kỹ thuật *cross-fade / overlap-add (OLA)*; phân tích định lượng sự khác biệt trên miền thời gian (waveform) và miền tần số (spectrogram).
5. **Ngữ điệu & Mô hình Source–Filter**: Giải thích vai trò của tần số cơ bản ($F_0$), thời lượng (duration), năng lượng (energy) và mô hình nguồn–lọc (source–filter) trong việc điều khiển cao độ, tốc độ nói mà không làm biến dạng bao hình phổ (spectral envelope).
6. **Tổng hợp tham số HMM**: Giải thích cơ chế HMM vận hành ở chế độ phát sinh (generation mode) để sinh chuỗi tham số âm học trơn theo trạng thái (state) và thời gian nhờ ràng buộc đặc trưng động ($\Delta$).
7. **TTS Học sâu (Neural TTS)**: Chạy suy luận (inference) với mô hình pretrained offline và phân tích quy trình `Text → Acoustic Representation (Mel-spectrogram) → Vocoder → Waveform`.
8. **Đánh giá & Thẩm định**: Đo lường *Intelligibility*, *Naturalness*, *Continuity*, *Latency* và dung lượng bộ nhớ để đối sánh toàn diện 3 thế hệ công nghệ TTS.

## 2. Cơ sở Lý thuyết, Mô hình Toán học và Công thức (Methods & Formulations)

### 2.1. Chuẩn hóa văn bản và Biểu diễn phát âm (Front-End Processing)
Quá trình biến đổi từ văn bản thô $T$ sang chuỗi đơn vị phát âm $P$:

$$T \xrightarrow{\text{normalize}} W = (w_1, w_2, \dots, w_K) \xrightarrow{\text{G2P / Lexicon}} P = (p_1, p_2, \dots, p_L)$$

- $T$: Văn bản đầu vào chứa chữ hoa, dấu câu, ký hiệu số.
- $W$: Chuỗi từ (token) đã được chuẩn hóa về dạng "có thể đọc" dựa vào ngữ cảnh (ví dụ trong miền thông báo xếp hàng, mã số `"23"` được chuẩn hóa thành `"hai ba"` thay vì `"hai mươi ba"`).
- $P$: Chuỗi âm vị (phone) hoặc đơn vị phát âm. Trong bài toán miền hẹp sử dụng trực tiếp đơn vị từ (**word-unit**), bước G2P có thể ánh xạ trực tiếp $W \to P$.

### 2.2. Mô hình Ngữ điệu (Prosody) và Biến đổi Cao độ
Mỗi đơn vị phát âm được đặc trưng bởi bộ tham số ngữ điệu $\{d, F_0(t), A(t)\}$, gồm thời lượng $d$, đường cong tần số cơ bản $F_0(t)$ và đường cong biên độ/năng lượng $A(t)$.

- **Tần số cơ bản ($F_0$) của âm hữu thanh**: Với tín hiệu lấy mẫu ở tần số $F_s$ và chu kỳ pitch tương ứng $N_p$ mẫu:
  $$F_0 = \frac{F_s}{N_p}$$
  *(Ví dụ: $F_s = 16\text{ kHz}, N_p = 100\text{ mẫu} \Rightarrow F_0 = 160\text{ Hz}$. Khi $N_p$ giảm thì $F_0$ tăng, giọng nghe cao hơn).*

- **Quan hệ dịch cao độ theo Semitone**: Khi dịch cao độ một khoảng $\Delta s$ semitone, tỷ số tần số $r$ và tần số cơ bản mới $F_{0,\text{new}}$ được tính bởi:
  $$r = 2^{\Delta s / 12}, \quad F_{0,\text{new}} = r \cdot F_{0,\text{old}}$$
  *(Ví dụ: $\Delta s = +12\text{ semitone} \Rightarrow r = 2$, tương ứng tăng đúng 1 quãng tám / octave).*

### 2.3. Mô hình Nguồn–Lọc (Source–Filter Model)
Theo Rabiner–Schafer, tín hiệu tiếng nói $X(z)$ được tạo thành từ nguồn kích thích $E(z)$ đi qua bộ lọc tuyến âm (vocal-tract filter) $H(z)$:

$$X(z) = E(z)H(z), \quad H(z) = \frac{B(z)}{A(z)} = \frac{b_0 + \sum_{k} b_k z^{-k}}{1 - \sum_{k} a_k z^{-k}}$$

- **Nguồn kích thích $e[n]$**: Gần tuần hoàn (chuỗi xung theo chu kỳ pitch) đối với âm hữu thanh (*voiced*), hoặc nhiễu ngẫu nhiên đối với âm vô thanh (*unvoiced*).
- **Bộ lọc $H(z)$**: Các hệ số lọc biến thiên theo thời gian quy định bao hình phổ (*spectral envelope*) và tần số cộng hưởng (*formant*).

### 2.4. Tổng hợp Ghép nối và Thuật toán Unit Selection
Với chuỗi $J$ vị trí mục tiêu $(t_1, t_2, \dots, t_J)$, bài toán chọn chuỗi candidate $\theta^* = (\theta_1, \dots, \theta_J)$ từ kho dữ liệu (inventory) nhằm tối thiểu hóa tổng chi phí:

$$\theta^* = \arg\min_{\theta_1 \dots \theta_J} \left[ \sum_{j=1}^{J} C_{\text{target}}(\theta_j, t_j) + \lambda \sum_{j=1}^{J-1} C_{\text{join}}(\theta_j, \theta_{j+1}) \right]$$

- **Target Cost $C_{\text{target}}(\theta_j, t_j)$**: Đo độ sai lệch giữa candidate $\theta_j$ và đặc tính mong muốn tại vị trí $t_j$ (về duration, RMS energy, ngữ cảnh).
- **Join Cost $C_{\text{join}}(\theta_j, \theta_{j+1})$**: Đo độ bất liên tục ("độ êm") khi ghép nối đuôi của $\theta_j$ với đầu của $\theta_{j+1}$.
- **Hệ số $\lambda$**: Trọng số điều chỉnh sự cân bằng giữa độ khớp mục tiêu và độ mượt tại biên nối.

### 2.5. Kỹ thuật Làm mượt Biên nối (Cross-Fade, OLA và TD-PSOLA)
- **Tuyến tính hóa biên nối (Linear Cross-fade)**: Trên vùng chồng lấn gồm $N$ mẫu với hàm dốc $\alpha[n] \in [0, 1]$:
  $$y[n] = (1 - \alpha[n])x_1[n] + \alpha[n]x_2[n], \quad 0 \le n < N$$
  Độ dài vùng chồng lấn $10\text{–}30\text{ ms}$ giúp triệt tiêu hiện tượng lệch pha/biên độ gây tiếng click mà không làm nhòe phụ âm chuyển tiếp.

- **Nguyên lý Overlap-Add (OLA)**: Các khung tín hiệu (frame) sau khi nhân cửa sổ $w[n]$ được dịch theo bước nhảy $R$ và cộng dồn:
  $$y[n] = \sum_{r} x_r[n] = x[n] \sum_{r} w[rR - n]$$
  Nếu tổng các cửa sổ chồng lấn thỏa mãn $\sum_r w[rR - n] = \text{const}$, tín hiệu được tái tạo hoàn hảo không méo biên độ.

- **TD-PSOLA (Time-Domain Pitch-Synchronous Overlap-Add)**: Tách các đoạn tín hiệu xung quanh các thời điểm kích thích pitch (pitch epochs) cách nhau chu kỳ gốc $T_0$:
  $$s_i[n] = s[n] w[n - iT_0]$$
  Khi dịch các phân đoạn $s_i[n]$ sang khoảng cách chu kỳ mới $T$ rồi thực hiện overlap-add, chu kỳ pitch đầu ra đổi thành $T$ trong khi bao hình phổ (formant) vẫn được bảo toàn.

### 2.6. Tổng hợp Tham số dựa trên HMM (HMM-based Parametric Synthesis)
Khác với bài toán nhận dạng, HMM trong TTS hoạt động ở chế độ sinh (**generation mode**): với mô hình $\lambda = (A, B, \pi)$ và chuỗi trạng thái (state sequence) được xác định từ chuỗi âm vị/ngữ cảnh, hệ thống tìm chuỗi vector tham số âm học tối ưu $\hat{O}^*$ theo tiêu chuẩn hợp lý cực đại (Maximum Likelihood):

$$\lambda = (A, B, \pi), \quad \hat{O}_t = [c_t, \Delta c_t], \quad \hat{O}^* = \arg\max_{O} P(O \mid \lambda, \text{state sequence})$$

- $c_t$: Vector đặc trưng tĩnh tại frame $t$ (bao hình phổ, formant, hoặc MFCC-like features).
- $\Delta c_t$: Đặc trưng động (tốc độ biến thiên giữa các frame kế tiếp), đóng vai trò ràng buộc toán học giúp quỹ đạo tham số sinh ra chuyển tiếp mượt mà giữa các trạng thái HMM thay vì nhảy bậc thang.

## 3. Hệ thống Chỉ số Đánh giá (Metrics Definition)

Các chỉ số định lượng và định tính bắt buộc sử dụng trong bài Lab để đánh giá chất lượng âm thanh và hiệu năng hệ thống:

| Nhóm chỉ số | Tên Metric | Công thức / Định nghĩa chuẩn | Mục đích & Ý nghĩa đánh giá |
| :--- | :--- | :--- | :--- |
| **Đặc trưng tín hiệu** | **RMS Energy** | $\text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x^2[n] + 10^{-12}}$ | Đo mức năng lượng hiệu dụng của unit; dùng để chuẩn hóa âm lượng (`target = 0.08`) trước khi ghép nối. |
| **Chi phí ghép nối** | **Normalized Feature Join Cost ($C^c_k$)** | $C^c_k(\theta_j, \theta_{j+1}) = \frac{\left[f^{\theta_j}_k(m) - f^{\theta_{j+1}}_k(1)\right]^2}{\sigma_k^2}$ | Đo độ lệch chuẩn hóa của đặc trưng thứ $k$ (log-energy, $F_0$, phổ) giữa frame cuối ($m$) của $\theta_j$ và frame đầu ($1$) của $\theta_{j+1}$ chia cho phương sai tự nhiên $\sigma_k^2$. |
| **Chi phí phổ biên** | **Spectral Discontinuity ($D_{\text{spec}}$)** | $D_{\text{spec}} = \frac{1}{K}\sum_{k=1}^{K} \left(L_{\text{left}}[k] - L_{\text{right}}[k]\right)^2$ | Sai số bình phương trung bình (MSE) trên phổ biên độ loga (dB): $L = 20\log_{10}(\max(\|\text{RFFT}\|, 10^{-8}))$ tính trên cửa sổ Hanning $20\text{–}30\text{ ms}$ hai phía biên nối. |
| **Chất lượng chủ quan** | **Mean Opinion Score (MOS)** | $\text{MOS} = \frac{1}{K}\sum_{i=1}^{K} r_i, \quad r_i \in \{1, 2, 3, 4, 5\}$ | Điểm đánh giá chủ quan trung bình trên $K$ lượt nghe (listening test mù từ 5–10 người). Cần tách riêng điểm *Naturalness* và *Intelligibility*. |
| **Độ rõ nội dung** | **Intelligibility (%)** | $\text{Intelligibility (\%)} = 100 \times \frac{N_{\text{correct}}}{N_{\text{total}}}$ | Tỷ lệ phần trăm số từ hoặc số câu người nghe nhận diện chính xác so với tổng số từ/câu kiểm thử. |
| **Độ tự nhiên** | **Naturalness** | Thang điểm 1–5 hoặc so sánh cặp mù A/B | Đánh giá mức độ giống giọng người thật, ngữ điệu mượt mà, ít âm hưởng máy móc (robotic/buzzy). |
| **Độ liên tục** | **Continuity** | Quan sát spectrogram & nghe mù tại biên unit | Đánh giá mức độ triệt tiêu tiếng click, độ êm năng lượng, không bị nhảy vọt pitch hay đứt gãy phổ tại ranh giới từ. |
| **Hiệu năng hệ thống** | **Latency & Footprint** | $T_{\text{latency}} = t_{\text{audio\_ready}} - t_{\text{text\_input}}$ (ms/s); Dung lượng (MB) | Thời gian trễ từ lúc nhận chuỗi văn bản đến khi sinh xong toàn bộ waveform trên cùng cấu hình máy, kèm kích thước lưu trữ của inventory/model. |

## 4. Yêu cầu Thực hành Bắt buộc & Hướng dẫn Triển khai (Implementation Guide)

### 4.1. Phần A — Text Normalization và Xây dựng Inventory
1. **Xây dựng hàm `normalize_text()` / `normalize_queue_text()`**:
   - Chuyển toàn bộ chuỗi đầu vào về chữ thường (`lowercase`), loại bỏ dấu chấm, dấu phẩy và các ký tự đặc biệt không cần thiết.
   - Tách token theo khoảng trắng; mọi chuỗi chữ số (`tok.isdigit()`) trong miền thông báo xếp hàng được tách và ánh xạ đọc từng chữ số từ `"0"` (`"không"`) đến `"9"` (`"chín"`).
   - Ví dụ chuẩn: `"Mời khách hàng số 23 đến quầy 5."` $\to$ `["mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"]`.
2. **Thu thập và chuẩn hóa Inventory**:
   - Tạo tối thiểu **15 từ/đơn vị** từ vựng miền hẹp, mỗi đơn vị có **2–3 bản ghi candidate** (ví dụ: `moi_01.wav`, `moi_02.wav`).
   - Thu âm cùng microphone, cùng khoảng cách, tần số lấy mẫu $F_s = 16\text{ kHz}$, 1 kênh (mono), 16-bit PCM. Giữ $100\text{–}200\text{ ms}$ khoảng lặng ngữ cảnh khi thu rồi cắt gọn hợp lý.
   - Chuẩn hóa RMS vừa phải (tránh normalize quá gắt làm mất sắc thái tự nhiên nhưng phải chống tràn biên `np.clip(y, -0.99, 0.99)`).
   - Lưu bảng `metadata.csv` gồm các trường: `label`, `speaker`, `duration`, `rms`, `file_path`.
   - Vẽ biểu đồ **Histogram phân bố duration và RMS** của toàn bộ inventory để phát hiện các mẫu bất thường (outliers).

### 4.2. Phần B — Baseline Concatenation
1. Chọn cố định 1 candidate (ví dụ bản `_01`) cho mỗi token trong câu mục tiêu và ghép nối trực tiếp không làm mượt (`xfade_ms = 0`).
2. Xuất kết quả ra file `audio_output/raw_concat.wav`.
3. Vẽ đồ thị **Waveform và Spectrogram** của câu ghép nối, kẻ vạch đánh dấu chính xác các thời điểm biên nối giữa các từ.
4. Nghe thẩm định và ghi nhận chi tiết các hiện tượng méo tại biên: tiếng nổ *click*, lệch năng lượng (*energy mismatch*), nhảy cao độ (*pitch jump*) hoặc đứt gãy bao hình phổ (*spectral discontinuity*).

### 4.3. Phần C — Unit Selection và Boundary Smoothing
1. **Thiết kế hàm chi phí**:
   - Tính **Target Cost** đơn giản dựa trên độ lệch duration và RMS của candidate so với mức trung bình/mục tiêu.
   - Tính **Join Cost** trên cửa sổ $20\text{–}30\text{ ms}$ (mặc định `ms = 25`, `nfft = 512`, cửa sổ Hanning) ở đuôi unit đứng trước và đầu unit đứng sau theo công thức $D_{\text{spec}}$ (có thể cộng thêm độ lệch $|\text{RMS}_{\text{left}} - \text{RMS}_{\text{right}}|$ và $|F_{0,\text{left}} - F_{0,\text{right}}|$ đã chuẩn hóa):
   ```python
   def join_cost(a, b, sr=16000, ms=25, nfft=512):
       n = int(sr * ms / 1000)
       wa = np.hanning(n)
       wb = np.hanning(n)
       A = np.fft.rfft(a[-n:] * wa, n=nfft)
       B = np.fft.rfft(b[:n] * wb, n=nfft)
       La = 20 * np.log10(np.maximum(np.abs(A), 1e-8))
       Lb = 20 * np.log10(np.maximum(np.abs(B), 1e-8))
       return float(np.mean((La - Lb) ** 2))
   ```
2. **Tìm kiếm chuỗi Candidate tối ưu**:
   - Áp dụng **Quy hoạch động (Dynamic Programming / Viterbi)** hoặc duyệt toàn bộ (exhaustive search với câu ngắn) để chọn tổ hợp candidate có tổng chi phí thấp nhất. Xuất bảng so sánh tổng chi phí giữa ít nhất 2 phương án chọn candidate.
3. **Thực nghiệm làm mượt Cross-fade**:
   - Cài đặt hàm chuẩn hóa năng lượng `match_rms(x, target=0.08)` và hàm trộn tuyến tính `crossfade(a, b, n)`:
   ```python
   def crossfade(a, b, n):
       n = min(n, len(a), len(b))
       if n == 0:
           return np.concatenate([a, b])
       alpha = np.linspace(0.0, 1.0, n, endpoint=True)
       mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
       return np.concatenate([a[:-n], mid, b[n:]])
   ```
   - Thử nghiệm với 4 mức độ dài cross-fade: **0 ms, 5 ms, 15 ms, 30 ms**. Chọn tham số tối ưu dựa trên kết quả nghe mù và độ giảm $D_{\text{spec}}$ tại ít nhất 2 biên nối, sau đó lưu file `audio_output/best_concat.wav`.

### 4.4. Phần D — Prosody và Mô hình Source–Filter
1. Chọn ít nhất 1 đơn vị âm hữu thanh (voiced unit), trích xuất một đoạn tín hiệu gần tuần hoàn, đếm số mẫu trong một chu kỳ cơ bản $N_p$ và kiểm chứng thực nghiệm công thức $F_0 \approx F_s / N_p$.
2. Thực hiện biến đổi tốc độ/thời lượng (**time-stretch / WSOLA** ở các mức `0.8x`, `1.0x`, `1.2x`) và dịch cao độ (**pitch shift** ở các mức `-2`, `0`, `+2` semitone).
3. Vẽ và đối chiếu Spectrogram trước/sau khi biến đổi; giải thích dựa trên mô hình source–filter vì sao khi thay đổi duration và pitch ta cần giữ nguyên bao hình phổ (spectral envelope / formant) để tránh hiệu ứng giọng biến dạng ("chipmunk").

### 4.5. Phần E — Tổng hợp Tham số dựa trên HMM (Guided Observation)
1. Gọi hàm từ backend HMM đã chuẩn bị sẵn:
   ```python
   from hmm_tts_backend import synthesize_hmm
   y_hmm, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
   ```
2. Trích xuất chuỗi trạng thái (state sequence) và vẽ đồ thị ít nhất **2 quỹ đạo tham số theo frame**: đường cong tần số cơ bản (`tracks["f0"]`) và một tham số phổ/formant.
3. Can thiệp thay đổi 1 tham số điều khiển (ví dụ co giãn duration theo tỷ lệ `0.8 / 1.2` hoặc dịch $F_0$ theo tỷ lệ `0.9 / 1.1`), tái tổng hợp âm thanh (nếu backend hỗ trợ) và lưu `audio_output/hmm_tts.wav`.
4. Phân tích bản chất sự khác biệt giữa *"chọn đoạn waveform tự nhiên có sẵn"* (Unit Selection) và *"sinh quỹ đạo tham số liên tục từ phân bố thống kê của trạng thái HMM"*.

### 4.6. Phần F — TTS Học sâu với Mô hình Pretrained Offline
1. Sử dụng wrapper offline `neural_tts_backend` để tổng hợp cùng **3 câu kiểm thử (test sentences)** như ở phương pháp ghép nối:
   ```python
   from neural_tts_backend import synthesize
   y_nn, sr, extra = synthesize("Mời khách hàng số 23 đến quầy 5")
   sf.write("audio_output/neural_tts.wav", y_nn, sr)
   ```
2. Lưu file waveform đầu ra và vẽ biểu đồ **Mel-spectrogram** trung gian (nếu trả về trong `extra`).
3. Đo đạc chính xác thời gian trễ (**latency**) từ lúc truyền chuỗi văn bản đến khi nhận xong mảng âm thanh trên cùng thiết bị phần cứng.

### 4.7. Phần G — Đánh giá Tổng hợp và So sánh
Tổ chức đánh giá chéo trên ít nhất 3 câu kiểm thử (chuẩn hóa cùng mức âm lượng loudness trước khi nghe mù) và điền đầy đủ dữ liệu thực nghiệm vào bảng tổng hợp:

| Phương pháp | Intelligibility (%) | Naturalness (MOS 1–5) | Continuity (Đánh giá biên nối) | Latency (ms) | Bộ nhớ / Dữ liệu (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Concatenation / Unit Selection** | *Đo từ listening test* | *Điểm MOS / A-B test* | *Độ êm sau cross-fade, $D_{\text{spec}}$* | *Thời gian ghép nối* | *Dung lượng thư mục `units/`* |
| **HMM / Parametric** | *Đo từ listening test* | *Điểm MOS / A-B test* | *Độ trơn quỹ đạo tham số ($\Delta$)* | *Thời gian sinh tham số + vocoder* | *Dung lượng state-parameter files* |
| **Neural TTS (Pretrained)** | *Đo từ listening test* | *Điểm MOS / A-B test* | *Độ liền mạch mel-spectrogram* | *Thời gian inference end-to-end* | *Dung lượng trọng số model* |

## 5. Thiết kế 5 Thí nghiệm Bắt buộc & Câu hỏi Báo cáo

### 5.1. Ma trận 5 thí nghiệm bắt buộc
Sinh viên phải thực hiện đầy đủ 5 thí nghiệm có kiểm soát dưới đây và trình bày kết quả vào báo cáo:

| TN | Biến thay đổi | Tập giá trị khảo sát | Câu hỏi nghiên cứu cần trả lời bằng thực nghiệm |
| :---: | :--- | :--- | :--- |
| **1** | **Cross-fade length** | `0, 5, 15, 30 ms` | Mức độ dài cross-fade nào triệt tiêu tiếng click hiệu quả nhất mà ít làm mờ phụ âm chuyển tiếp nhất? |
| **2** | **Candidate / unit** | `1` candidate cố định so với `2–3` candidate | Thuật toán Unit Selection có làm giảm rõ rệt join cost ($D_{\text{spec}}$) và cải thiện độ tự nhiên (naturalness) không? |
| **3** | **Duration / Speed** | `0.8×, 1.0×, 1.2×` | Việc co giãn tốc độ nói ảnh hưởng như thế nào đến độ rõ (intelligibility) và độ tự nhiên (naturalness)? |
| **4** | **Pitch Shift** | `-2, 0, +2 semitone` | Dịch cao độ có làm xuất hiện nhiễu nhân tạo (artifact) hay hiện tượng rè giọng (buzziness) không? |
| **5** | **Synthesis Backend** | `Concat / HMM / Neural` | Phương pháp tổng hợp nào vượt trội nhất xét theo từng tiêu chí riêng biệt (độ rõ, độ tự nhiên, độ trễ, tài nguyên)? |

### 5.2. Mười câu hỏi lý thuyết & phân tích trong báo cáo
Báo cáo (độ dài 5–8 trang) phải giải đáp ngắn gọn, chính xác 10 câu hỏi chuyên môn sau:
1. Vì sao một câu được ghép từ các đoạn thu âm "tự nhiên" vẫn có thể nghe kém tự nhiên khi phát cả câu?
2. Phân biệt *target cost* và *join cost*. Nếu chỉ tối ưu hóa *join cost* (cho $\text{target cost} = 0$) thì có thể xảy ra vấn đề gì?
3. Vì sao thiết lập vùng *cross-fade* quá dài có thể làm giảm độ rõ của các âm phụ âm?
4. Giải thích bằng mô hình *source–filter*: vì sao thay đổi tần số cơ bản $F_0$ không đồng nghĩa với việc thay đổi tần số *formant*?
5. Kỹ thuật *PSOLA* thay đổi pitch bằng cơ chế nào? Tại sao các âm xát hữu thanh (*voiced fricatives*) dễ bị rè (*buzziness*) khi kéo dài thời lượng quá mức?
6. Trong tổng hợp tham số *HMM-based synthesis*, vì sao việc bổ sung đặc trưng động (*delta feature* $\Delta c_t$) lại giúp quỹ đạo tham số sinh ra trơn tru hơn?
7. Tại sao phương pháp ghép nối nguyên dạng sóng (*whole-waveform concatenation*) đòi hỏi kho dữ liệu (*inventory*) rất lớn mới đạt được độ tự nhiên cao?
8. Nêu ít nhất hai điểm khác biệt cơ bản về mặt nguyên lý giữa *Concatenative TTS* và *Neural TTS*.
9. Vì sao không nên đánh giá, so sánh *Neural TTS* bằng một mô hình tùy ý tải ngẫu nhiên từ Internet ngay trong buổi Lab?
10. Nếu một hệ thống TTS đạt điểm *MOS (Naturalness)* rất cao nhưng *Intelligibility* lại thấp thì hệ thống đó có thể sử dụng trong ứng dụng thực tế nào? Giải thích.

## 6. Danh mục Sản phẩm Nộp, Kiểm thử Lỗi & Rubric Chấm điểm

### 6.1. Yêu cầu sản phẩm nộp tối thiểu (Deliverables Checklist)
- **Mã nguồn thực thi**: `Lab04.ipynb` hoặc `Lab04.py` chạy lại được từ đầu (end-to-end reproducible) với cấu trúc thư mục rõ ràng.
- **Thư mục âm thanh (`audio_output/`)**: Chứa đầy đủ `raw_concat.wav`, `best_concat.wav`, `hmm_tts.wav` (nếu backend hỗ trợ xuất audio) và `neural_tts.wav`.
- **Bộ biểu đồ phân tích (tối thiểu 5 hình)**:
  1. Thống kê phân bố duration & RMS của inventory (`inventory stats`).
  2. Waveform & Spectrogram tại biên ghép nối trước và sau khi áp dụng cross-fade.
  3. Bảng/đồ thị so sánh Join Cost & Total Cost giữa ít nhất 2 phương án chọn candidate trong Unit Selection.
  4. Biểu đồ quỹ đạo tham số $F_0$ contour và spectral/formant track từ backend HMM.
  5. Biểu đồ/bảng tổng hợp so sánh đầu ra giữa 3 phương pháp (kèm latency và MOS/A-B test cho ít nhất 3 câu).
- **Dữ liệu khảo sát**: File `ratings.csv` lưu kết quả đánh giá nếu nhóm tổ chức listening test.
- **Báo cáo kỹ thuật**: Độ dài 5–8 trang trình bày phương pháp, công thức, thiết kế thí nghiệm, phân tích kết quả nghe chủ quan, trả lời 10 câu hỏi và rút ra kết luận.

### 6.2. Bảng chẩn đoán lỗi thường gặp và cách khắc phục (Troubleshooting)

| Hiện tượng lỗi | Nguyên nhân gốc rễ | Cách kiểm tra và khắc phục chuẩn |
| :--- | :--- | :--- |
| **Tiếng nổ (click) tại biên nối** | Lệch thành phần một chiều (DC offset), chênh lệch RMS hoặc ngược pha tại điểm cắt | Cắt biên tại điểm qua không (zero-crossing) tương đối; chuẩn hóa `match_rms()`; áp dụng `crossfade()` 10–30 ms. |
| **Âm lượng bị nhỏ đi hoặc dồn cục ở biên** | Công thức cộng/trừ hàm dốc $\alpha[n]$ trong cross-fade bị sai | Kiểm tra lại mảng $\alpha$ tăng tuyến tính từ $0 \to 1$, tổng trọng số $(1-\alpha) + \alpha = 1$ và độ dài đoạn chồng lấn $n$. |
| **Câu phát ra sai nội dung chữ/số** | Lỗi hàm `normalize_text()` hoặc ánh xạ sai tên file token | In chuỗi token (`print(tokens)`) ngay sau bước chuẩn hóa để đối chiếu trước khi gọi hàm ghép nối. |
| **Dịch pitch nghe như giọng hoạt hình ("chipmunk")** | Thay đổi đồng thời cả pitch lẫn bao hình phổ (spectral envelope / formant) do co giãn trục thời gian trực tiếp | Sử dụng thuật toán tách nguồn–lọc (TD-PSOLA hoặc vocoder) để giữ nguyên vị trí formant khi dịch $F_0$. |
| **Quỹ đạo tham số HMM bị giật bậc thang** | Không đưa đặc trưng động ($\Delta$-feature) vào quá trình tối ưu hoặc sai tỷ lệ frame rate | Kiểm tra lại ràng buộc $\Delta c_t$ trong hàm sinh tham số và tốc độ khung hình (frame shift 10 ms). |
| **Neural TTS chạy khác máy ra kết quả khác nhau** | Khác phiên bản mô hình, thư viện hoặc thiết bị tính toán (CPU/GPU, random seed) | Đóng gói cố định trọng số model, phiên bản thư viện và seed trong wrapper offline trước giờ Lab. |
| **Kết quả listening test thiếu công bằng** | Đánh giá trên các câu văn bản khác nhau hoặc chênh lệch âm lượng lớn giữa 3 hệ thống | Chạy cùng một tập 3 câu kiểm thử, chuẩn hóa độ to (loudness/RMS) đồng nhất và ẩn danh tên phương pháp (blind test). |

### 6.3. Rubric chấm điểm chuẩn (Thang điểm 10.0)

| Hạng mục đánh giá | Trọng số | Tiêu chí nghiệm thu đạt điểm tối đa |
| :--- | :---: | :--- |
| **1. Chuẩn bị dữ liệu & Normalization** | **1.0 điểm** | Inventory đúng định dạng (16 kHz, mono, 16-bit PCM), âm thanh sạch, quy tắc `normalize_text()` xử lý chuẩn chuỗi số và từ vựng, có `metadata.csv` và biểu đồ thống kê. |
| **2. Concatenative Baseline** | **1.5 điểm** | Ghép nối và phát sinh đúng nội dung câu mục tiêu (`raw_concat.wav`), vẽ waveform + spectrogram đánh dấu đúng biên nối và giải thích rõ hiện tượng méo. |
| **3. Unit Selection & Smoothing** | **2.0 điểm** | Xây dựng hàm target/join cost hợp lý, chọn được chuỗi candidate tối ưu, cài đặt đúng cross-fade/OLA (`best_concat.wav`) và phân tích định lượng trước/sau. |
| **4. Prosody & Source–Filter** | **1.0 điểm** | Đo kiểm chứng $F_0 \approx F_s/N_p$ chính xác, thực hiện đầy đủ thí nghiệm thay đổi duration/pitch và giải thích đúng bản chất bảo toàn bao hình phổ. |
| **5. HMM-based Synthesis** | **1.5 điểm** | Trích xuất đúng chuỗi trạng thái, vẽ rõ quỹ đạo $F_0$ và tham số phổ/formant theo frame, thử nghiệm thay đổi tham số và giải thích đúng cơ chế HMM generation. |
| **6. Neural TTS Inference** | **1.0 điểm** | Chạy đúng mô hình pretrained offline trên 3 câu test, lưu file `neural_tts.wav`, trích xuất mel-spectrogram và đo chính xác latency. |
| **7. Đánh giá tổng hợp & Báo cáo** | **2.0 điểm** | Bảng so sánh 3 phương pháp có số liệu thực nghiệm đầy đủ, đủ tối thiểu 5 hình minh họa và file audio, trả lời sâu sắc các câu hỏi với kết luận có căn cứ khoa học. |

## 7. Tài liệu Tham khảo Chuẩn
1. **X. Huang, A. Acero, H.-W. Hon**, *Spoken Language Processing: A Guide to Theory, Algorithm, and System Development*, Prentice Hall, 2001 — Chương 14 (*Text and Phonetic Analysis*), Chương 15 (*Prosody*), Chương 16 (*Speech Synthesis*).
2. **L. R. Rabiner, R. W. Schafer**, *Theory and Applications of Digital Speech Processing*, Prentice Hall, 2010 — Chương 13 (*Text-to-Speech Synthesis Methods*: mô hình source–filter, unit selection và TD-PSOLA).
