**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

**TRƯỜNG ĐẠI HỌC THỦY LỢI** KHOA CÔNG NGHỆ THÔNG TIN • BỘ MÔN TRÍ TUỆ NHÂN TẠO 

**CSE457 XỬ LÝ ÂM THANH VÀ TIẾNG NÓI** 

# **LAB 2** 

**ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG BẰNG DTW** Từ phân tích ngắn hạn đến MFCC, căn chỉnh thời gian động và nhận dạng từ đơn 

|**Thông tin**|**Nội dung**|
|---|---|
|Thời lượng|03 tiết thực hành|
|Hình thức|Cá nhân, lập trình trên Jupyter/VS Code|
|Ngôn ngữ khuyến nghị|Python 3.x (NumPy, SciPy, Matplotlib, Librosa, SoundFile,<br>scikit-learn)|
|Đầu ra|Notebook + bộ dữ liệu ghi âm + hình minh họa + ma trận nhầm<br>lẫn + demo nhận dạng|
|Liên kết đề cương|Chương 2: 2.1 Nhận dạng mẫu; 2.2 Các đặc trưng cơ bản của<br>tiếng nói; 2.3 Nhận dạng bằng đối sánh mẫu|



Trang 1 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

## **1. Mục tiêu và chuẩn đầu ra của Lab** 

Lab 2 biến các khái niệm của Chương 2 thành một hệ nhận dạng từ đơn có thể chạy được. Trọng tâm là hiểu bản chất của đặc trưng tiếng nói và tự cài đặt phép đối sánh DTW thay vì chỉ gọi một API nhận dạng hoàn chỉnh. 

- Phân tích tín hiệu tiếng nói theo frame; tính short-time energy, magnitude và zero-crossing rate (ZCR). 

- Dùng energy/ZCR để xác định vùng có tiếng nói và cắt bỏ silence ở đầu/cuối. 

- Tính short-time autocorrelation và liên hệ đỉnh tương quan với tính tuần hoàn/pitch. 

- Trích chọn vector MFCC từ từng frame và giải thích vai trò của thang Mel, filterbank, log và DCT. 

- Xây dựng hàm khoảng cách giữa hai vector đặc trưng và ma trận local distance giữa hai chuỗi MFCC. 

- Tự cài đặt Dynamic Time Warping (DTW), truy vết đường căn chỉnh tối ưu và chuẩn hóa chi phí. 

- Xây dựng bộ nhận dạng 5–10 từ tách rời theo nearest-template; đánh giá bằng accuracy và confusion matrix. 

**Kết quả đầu ra bắt buộc:** Một chương trình nhận dạng từ đơn có thể nhận file WAV chưa biết, trích MFCC, so sánh DTW với các template đã lưu và trả về nhãn dự đoán cùng khoảng cách DTW. 

## **1.1. Phân bổ 3 tiết** 

|**Tiết**|**Trọng tâm**|**Kết quả trung gian**|
|---|---|---|
|1|Dữ liệu, framing, energy, ZCR,<br>endpoint detection|Waveform + energy/ZCR + đoạn<br>speech đã cắt|
|2|MFCC, khoảng cách vector, local-<br>distance matrix|Ma trận MFCC + minh họa Mel<br>filterbank + distance matrix|
|3|DTW, nhận dạng nhiều template,<br>đánh giá|Recognizer + optimal path +<br>confusion matrix + accuracy|



## **1.2. Kiến thức và kỹ năng tiên quyết** 

- Lab 1: sampling rate, framing/windowing, FFT/STFT và biểu diễn dB. 

- Python cơ bản, NumPy array, vòng lặp, hàm, đọc/ghi WAV. 

- Khái niệm vector, khoảng cách Euclid và ma trận. 

- Đã đọc Chương 2 của bài giảng trước buổi Lab. 

**2. Chuẩn bị môi trường và dữ liệu** 

## **2.1. Môi trường Python** 

<mark>pip install numpy scipy matplotlib librosa soundfile scikit-learn</mark> 

**Lưu ý môi trường:** Không sử dụng dịch vụ ASR trực tuyến hoặc mô hình end-to-end trong bài Lab này. Mục tiêu là tự quan sát và cài đặt pipeline đặc trưng + DTW. 

Trang 2 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

## **2.2. Quy ước dữ liệu ghi âm** 

- Định dạng: WAV, mono, PCM; khuyến nghị Fₛ = 16 kHz. 

- Mỗi từ được nói tách rời, có khoảng 0.2–0.5 s silence trước và sau để phục vụ endpoint detection. 

- Giữ khoảng cách micro và mức âm lượng tương đối ổn định. 

- Tên thư mục nên dùng ASCII (khong, mot, hai, ba, bon) để tránh lỗi đường dẫn; nội dung nhãn vẫn là tiếng Việt. 

- Tối thiểu 5 lớp × 5 lần lặp = 25 file cho một người nói. Nếu làm nhóm 2 người, nên ghi đủ ở cả hai người để có thí nghiệm cross-speaker. 

<mark>dataset/ khong/khong_01.wav ... khong_05.wav mot/mot_01.wav       ... mot_05.wav hai/hai_01.wav       ... hai_05.wav ba/ba_01.wav         ... ba_05.wav bon/bon_01.wav       ... bon_05.wav</mark> 

**3. Tóm tắt kiến thức và công thức cần dùng** 



<!-- Start of picture text -->
QUE WAY _ detectionEndpoint _ HammingFrame + a MFCC _ templates]DTW véi —* — | nhandangTir duge<br><!-- End of picture text -->

_Hình 1. Pipeline của bộ nhận dạng từ đơn bằng MFCC + DTW._ 

## **3.1. Phân tích ngắn hạn: frame, hop và cửa sổ** 

Tiếng nói biến đổi theo thời gian nhưng có thể xem là gần dừng trong những khoảng ngắn. Vì vậy ta chia tín hiệu x[n] thành các frame chồng lấn. Với thời lượng frame T_f và bước dịch T_h: 



L là số mẫu/frame; R là số mẫu dịch giữa hai frame liên tiếp. Trong Lab dùng T_f ≈ 25 ms và T_h ≈ 10 ms. Cửa sổ Hamming: 





Ví dụ nhanh:  Với Fₛ = 16 kHz, frame 25 ms có L = 400 mẫu; hop 10 ms có R = 160 mẫu. Hai frame liên tiếp chồng lấn 240 mẫu, tương đương 15 ms. 

## **3.2. Pre-emphasis (khuyến nghị trước MFCC)** 

Một bộ lọc sai phân bậc một thường được dùng để tăng tương đối thành phần tần số cao trước phân tích phổ. Tài liệu Rabiner–Schafer minh họa phép tiền nhấn tương đương với lọc bởi δ[n] − 0.97δ[n−1]. 

Trang 3 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

y[n] = x[n] − αx[n−1],        α ≈ 0.97 

Trong Lab, phải dùng cùng một α cho toàn bộ training và test. Không tiền nhấn một phía rồi bỏ ở phía còn lại. 

Ví dụ nhanh:  Nếu x[n]=[0.00, 0.10, 0.18, 0.20], α=0.97 thì y[1]=0.10; y[2]≈0.083; y[3]≈0.025. Bộ lọc nhấn mạnh biến đổi nhanh hơn là mức DC/chậm. 

## **3.3. Short-time energy, magnitude và RMS** 

Với frame đã được cửa sổ hóa x_r[n], các đại lượng mức biên độ cơ bản là: 







Energy nhấn mạnh các mẫu biên độ lớn hơn vì bình phương; magnitude có dynamic range nhỏ hơn. Trong endpoint detection có thể dùng log-energy: 



Ví dụ nhanh:  Hai frame có cùng độ dài: frame A có RMS = 0.20, frame B có RMS = 0.02. Công suất trung bình của A lớn gấp (0.20/0.02)² = 100 lần, tức khoảng 20 dB. 

## **3.4. Zero-Crossing Rate (ZCR)** 

ZCR đếm mức độ thường xuyên mà tín hiệu đổi dấu. Với cửa sổ chữ nhật và frame dài L: 





Mỗi lần đổi dấu tạo trị tuyệt đối bằng 2, vì vậy hệ số 1/(2L) đưa kết quả về số zero-crossing trung bình trên một mẫu. Trong tiếng nói, voiced thường có năng lượng cao hơn và ZCR thấp hơn; unvoiced thường có ZCR cao hơn. 

**Ví dụ nhanh:** Nếu một frame 100 mẫu có 20 lần đổi dấu, ZCR = 20/100 = 0.20 crossing/sample. 

Trang 4 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 



<!-- Start of picture text -->
Vi du: tin higu co silence, voiced va unvoiced<br>0.50<br>0.25 |<br>cs<br>& 0.00<br>s<br>® 0.25 Hi Mi<br>0.50<br>°<br>3<br>>B -50<br>2<br>§<br>3@ 100<br>04<br>§<br>“02<br>0.0<br>0.0 0.2 04 06 os 10 12 14<br>Thai gian (s)<br><!-- End of picture text -->

_Hình 2. Ví dụ minh họa waveform, log-energy và ZCR trên tín hiệu mô phỏng._ 

## **3.5. Endpoint detection: tìm điểm bắt đầu/kết thúc từ nói** 

Đối với nhận dạng từ đơn, việc loại silence giúp DTW không phải căn chỉnh các đoạn nền không liên quan. Rabiner–Schafer mô tả endpoint detection dựa trên short-time log-energy kết hợp ZCR. Ý tưởng thực hành: 

- Ước lượng mức nền từ các frame silence đầu/cuối. 

- Dùng log-energy để tìm vùng speech thô. 

- Dùng ZCR để tinh chỉnh biên nếu từ bắt đầu/kết thúc bằng âm vô thanh hoặc fricative. 

- Giữ thêm vài frame đệm trước/sau biên để tránh cắt mất phụ âm. 

**Heuristic của Lab:** Để code ngắn gọn, case study dùng ngưỡng energy tương đối và một vùng đệm. Đây là lựa chọn triển khai của bài Lab, không phải một ngưỡng cố định trong giáo trình. Sinh viên phải báo cáo ngưỡng đã dùng. 

## **3.6. Short-time autocorrelation và pitch** 

Tự tương quan đo mức giống nhau của một frame khi dịch nó k mẫu. Dạng tổng quát trong Rabiner–Schafer: 

R_r[k] = Σₙ (x_r[n]) (x_r[n+k]) 

Trong frame hữu thanh, đỉnh tương quan đáng kể sau k=0 thường xuất hiện gần chu kỳ pitch N₀. Khi đó: 

F₀ ≈ Fₛ / N₀ 

**Ví dụ nhanh:** Nếu Fₛ = 16 kHz và đỉnh autocorrelation thứ nhất (sau lag 0) ở k = 100, ước lượng F₀ ≈ 16000/100 = 160 Hz. 

**Giới hạn sử dụng:** Pitch không phải đặc trưng bắt buộc để chạy DTW trong Lab này. Phần autocorrelation giúp sinh viên hiểu thêm tính tuần hoàn của speech và kiểm tra voiced/unvoiced; MFCC vẫn là feature chính cho nhận dạng. 

Trang 5 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

## **3.7. MFCC: Mel-Frequency Cepstral Coefficients** 

MFCC mô tả đường bao phổ ngắn hạn theo một thang tần số gần với cảm nhận thính giác. Pipeline: pre-emphasis → frame/window → FFT/power → Mel filterbank → log → DCT. 





Thang Mel trong Huang–Acero–Hon: 





Với M bộ lọc tam giác H_m[k], năng lượng log của bộ lọc m: 



MFCC thu được bằng DCT của các log-energy filterbank: 



Tài liệu nêu M thường trong khoảng 24–40 và nhiều hệ nhận dạng dùng 13 hệ số đầu. Trong Lab khuyến nghị M = 24, N_mfcc = 13. 



<!-- Start of picture text -->
Mel filterbank 24 b6 loc, Fs = 16 kHz<br>Fo \ /\<br>0.2 I / \<br>0 0OeoeoeoDODoHReeanananaaqoeee eee Sg)<br>1000 2000 3000 4000 5000 6000 7000 8000<br>Tansé (Hz)<br><!-- End of picture text -->

_Hình 3. Mel filterbank 24 bộ lọc; mật độ phân giải cao hơn ở vùng tần số thấp._ 

**Ví dụ nhanh:** Với một từ dài 0.8 s, frame 25 ms và hop 10 ms, ta thu xấp xỉ 78 frame. Nếu mỗi frame dùng 13 MFCC thì mẫu từ được biểu diễn bằng ma trận khoảng 13×78. 

## **3.8. Khoảng cách giữa hai vector đặc trưng** 

Với hai vector MFCC cùng số chiều, một local distance đơn giản là Euclidean distance: 

d(i,j) = ||x_i − y_j||₂ = √(Σ_q (x_i[q] − y_j[q])²) 

Rabiner–Schafer trình bày cepstral distance dựa trên Euclidean distance và liên hệ nó với sai khác giữa các log-spectrum đã làm trơn. 

**Ví dụ nhanh:** Nếu x=[1,2] và y=[4,6] thì d=√((1−4)²+(2−6)²)=5. 

Trang 6 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

## **3.9. Dynamic Time Warping (DTW)** 

Hai lần nói cùng một từ thường có số frame khác nhau do tốc độ nói khác nhau. DTW tìm một ánh xạ phi tuyến theo thời gian sao cho tổng local distance trên đường căn chỉnh là nhỏ nhất. 





Trong case study, dùng biến thể ba bước cục bộ (một lựa chọn triển khai của Lab): 

D[i,j] = C[i,j] + min{ D[i−1,j], D[i,j−1], D[i−1,j−1] } 

Điều kiện: bắt đầu tại (0,0), kết thúc tại (N−1,M−1), đường đi đơn điệu theo thời gian. Sau khi điền ma trận D, backtracking từ ô cuối cho optimal warping path. Để giảm thiên lệch theo chiều dài: 

DTW_norm(X,Y) = D[N−1,M−1] / |P| 

|P| là số cặp frame trên đường tối ưu. Giáo trình nhấn mạnh dynamic programming tránh phải liệt kê số lượng đường căn chỉnh tăng theo hàm mũ và DTW đặc biệt phù hợp nhận dạng từ vựng nhỏ. 



<!-- Start of picture text -->
Local-distance matrix va dudng DTW téi uu (vi du s6) 40<br>4 3.5<br>3.0<br>3<br>3 z<br>255<br>2a<br>s<br>8 S<br>32 2.0°8<br>& 152<br>1<br>1.0<br>0 0.5<br>0.0<br>0 1 2 3 4 5 6<br>Framej cla mau Y<br><!-- End of picture text -->

_Hình 4. Ma trận local distance và một optimal DTW path trong ví dụ số._ 

## **3.10. Luật quyết định và đánh giá** 

Với một test pattern X và tập template T_{w,r} của từ w, dùng khoảng cách tốt nhất trong các template: 

D_w(X) = min_r DTW_norm(X, T_{w,r}) 



Accuracy cho bộ test gồm N_test mẫu: 

Accuracy = N_correct / N_test × 100% 

Trang 7 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

Confusion matrix cho biết từ nào thường bị nhầm với từ nào, hữu ích hơn một con số accuracy duy nhất. 



<!-- Start of picture text -->
Vi du confusion matrix<br>khong 0) 0) 0) 0) 4.0<br>3.5<br>mot te) 1 0 te) 3.0<br>. 2.5<br>&é<br>c<br>S hai 0 ie} oO 0 2.0<br>2 15<br>ba 0 0 0 0 1.0<br>0.5<br>bon ie) 0 le) 1 0.0<br>oor ot ye we yo<br>Du doan<br><!-- End of picture text -->

_Hình 5. Ví dụ confusion matrix của bộ nhận dạng 5 từ._ 

## **4. Nội dung thực hành bắt buộc** 

Sinh viên thực hiện tuần tự A → G. Mỗi phần phải có code, ít nhất một hình hoặc bảng kết quả, và nhận xét kỹ thuật ngắn. 

## **A. Thu dữ liệu và kiểm tra chất lượng** 

- Tạo vocabulary 5–10 từ tách rời; mỗi từ tối thiểu 5 lần lặp. 

- Đưa tất cả file về WAV mono 16 kHz. 

- Vẽ waveform của ít nhất 3 từ; kiểm tra clipping và độ dài silence đầu/cuối. 

- Chia train/test trước khi chọn template, tránh dùng cùng file cho cả train và test. 

## **B. Đặc trưng miền thời gian** 

- Chia frame 25 ms, hop 10 ms. 

- Tính short-time energy/RMS và ZCR cho ít nhất 3 file. 

- Vẽ energy và ZCR theo thời gian trên cùng trục thời gian với waveform. 

- Mô tả khác biệt giữa silence, voiced và unvoiced. 

## **C. Endpoint detection** 

- Xây dựng một hàm tìm start/end frame dựa trên log-energy, có thể dùng ZCR để tinh chỉnh. 

- Xuất file speech đã trim và so sánh thời lượng trước/sau. 

- Kiểm tra ít nhất một từ bắt đầu/kết thúc bằng phụ âm năng lượng thấp; điều chỉnh margin để không cắt mất âm. 

## **D. MFCC** 

- Tiền nhấn α=0.97 (hoặc giải thích nếu không dùng). 

Trang 8 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

- Dùng Hamming window; n_mels=24; n_mfcc=13; giữ các tham số giống nhau cho train/test. 

- Vẽ ma trận MFCC của ít nhất 2 từ khác nhau. 

- Giải thích tại sao mỗi file có số frame khác nhau nhưng số chiều MFCC/frame là cố định. 

## **E. DTW** 

- Cài đặt Euclidean local distance. 

- Tự cài dynamic programming DTW; không chỉ gọi hàm thư viện trong phần bắt buộc. 

- Vẽ local-distance matrix và optimal path của: (i) cùng từ, (ii) hai từ khác nhau. 

- So sánh DTW_norm giữa hai trường hợp và giải thích. 

## **F. Bộ nhận dạng nearest-template** 

- Mỗi test file: trích feature → tính DTW đến toàn bộ templates → chọn nhãn có khoảng cách nhỏ nhất. 

- In bảng top-3 nhãn gần nhất và khoảng cách tương ứng cho ít nhất 5 test file. 

- Tùy chọn reject: nếu khoảng cách tốt nhất > θ, trả về 'unknown'; nếu dùng phải mô tả cách chọn θ. 

## **G. Đánh giá và thí nghiệm** 

- Tính accuracy và confusion matrix. 

- Thí nghiệm bắt buộc 1: có endpoint detection vs. không endpoint detection. 

- Thí nghiệm bắt buộc 2: MFCC 13 hệ số vs. MFCC + Δ (phần Δ có thể dùng thư viện). 

- Nếu có dữ liệu hai người: thêm thí nghiệm same-speaker vs. cross-speaker. 

## **4.1. Ma trận kết quả tối thiểu cần báo cáo** 

|**Thí nghiệm**|**Metric/kết quả phải có**|**Nhận xét bắt buộc**|
|---|---|---|
|Energy + ZCR|2–3 đồ thị theo thời gian|Silence/voiced/unvoiced thể<br>hiện thế nào?|
|Endpoint|Thời lượng trước/sau trim|Có cắt mất âm đầu/cuối<br>không?|
|MFCC|Heatmap của ≥2 từ|Cấu trúc theo thời gian khác<br>nhau ra sao?|
|DTW cùng từ|DTW_norm + path|Đường đi có gần đường chéo<br>không?|
|DTW khác từ|DTW_norm + path|Chi phí tăng thế nào?|
|Recognizer|Accuracy + confusion matrix|Cặp từ nào dễ nhầm nhất?|



Trang 9 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

**5. Case Study - Nhận dạng 5 từ tiếng Việt bằng MFCC + DTW** 

Case study này là một baseline hoàn chỉnh để sinh viên làm theo, sau đó thay dữ liệu/siêu tham số và phân tích kết quả của chính mình. 

## **5.1. Bài toán** 

Vocabulary: “không, một, hai, ba, bốn”. Mỗi từ ghi 5 lần. Chọn 3 file đầu làm template training và 2 file còn lại làm test. Đây là hệ speaker-dependent cơ bản; phần mở rộng cross-speaker thực hiện nếu làm nhóm. 

**Quy ước file:** Dùng nhãn ASCII trong đường dẫn: khong, mot, hai, ba, bon. Ghi chú mapping sang tiếng Việt trong báo cáo. 

## **5.2. Cấu hình tham số baseline** 

|**Tham số**|**Giá trị baseline**|**Ý nghĩa**|
|---|---|---|
|Fₛ|16,000 Hz|Chuẩn hóa tất cả file về cùng<br>sampling rate|
|Frame|25 ms = 400 mẫu|Giả thiết gần dừng|
|Hop|10 ms = 160 mẫu|Tốc độ frame 100 frame/s|
|Window|Hamming|Giảm spectral leakage|
|Pre-emphasis|α = 0.97|Nhấn tương đối tần số cao|
|NFFT|512|FFT mỗi frame|
|Mel filters|24|Trong khoảng thường dùng<br>theo tài liệu|
|MFCC|13 hệ số|Vector đặc trưng/frame|
|Local distance|Euclidean|Khoảng cách giữa hai frame<br>MFCC|
|DTW cost|Tổng / path length|Chuẩn hóa theo độ dài đường<br>căn chỉnh|



## **5.3. Bước 1 - Đọc, chuẩn hóa và trim endpoint** 

<mark>import numpy as np import librosa from scipy.signal import lfilter FS = 16000 FRAME_MS, HOP_MS = 25, 10 WIN = int(FS*FRAME_MS/1000)    # 400 HOP = int(FS*HOP_MS/1000)     # 160</mark> 

<mark>def load_audio(path): y, sr = librosa.load(path, sr=FS, mono=True) y = y / (np.max(np.abs(y)) + 1e-9) return y</mark> 

Trang 10 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

<mark>def trim_energy(y, top_db=35, margin_ms=50): # Heuristic baseline của Lab: ngưỡng energy tương đối. # Sinh viên phải thử nghiệm và báo cáo top_db đã dùng. yt, idx = librosa.effects.trim(y, top_db=top_db, frame_length=WIN, hop_length=HOP) m = int(FS*margin_ms/1000) s = max(0, idx[0]-m); e = min(len(y), idx[1]+m) return y[s:e], (s,e)</mark> 

**Yêu cầu học thuật:** Hàm trim_energy ở trên chỉ là baseline thực dụng. Trước khi dùng, sinh viên vẫn phải tự tính/vẽ short-time energy và ZCR để hiểu nguyên lý endpoint detection. 

## **5.4. Bước 2 - Trích MFCC** 

<mark>def mfcc_feature(y): # pre-emphasis y = lfilter([1.0, -0.97], [1.0], y)</mark> 

<mark>M = librosa.feature.mfcc( y=y, sr=FS, n_mfcc=13, n_mels=24, n_fft=512, win_length=WIN, hop_length=HOP, window='hamming', center=False ) # Cepstral mean normalization theo từng utterance M = M - np.mean(M, axis=1, keepdims=True) return M.T     # shape: (T frames, 13)</mark> 

CMN trong code là bước chuẩn hóa thực hành để giảm sai khác mức trung bình giữa utterance. Nếu giảng viên muốn baseline tối giản tuyệt đối, có thể bỏ CMN nhưng phải áp dụng nhất quán cho train và test. 

## **5.5. Bước 3 - Tự cài DTW** 

<mark>def dtw_distance(X, Y): # X: (N,D), Y: (M,D) N, M = len(X), len(Y) D = np.full((N+1, M+1), np.inf) D[0,0] = 0.0 back = np.zeros((N+1, M+1, 2), dtype=int)</mark> 

<mark>for i in range(1, N+1): for j in range(1, M+1): local = np.linalg.norm(X[i-1] - Y[j-1]) choices = [(D[i-1,j], i-1, j), (D[i,j-1], i, j-1), (D[i-1,j-1], i-1, j-1)] best, pi, pj = min(choices, key=lambda z: z[0]) D[i,j] = local + best back[i,j] = (pi,pj)</mark> 

Trang 11 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

<mark># backtrack path=[]; i,j=N,M while i>0 or j>0: path.append((i-1,j-1)) i,j = back[i,j] path.reverse() return D[N,M] / max(len(path),1), path, D[1:,1:]</mark> 

**Kiểm tra bắt buộc:** Với X và Y giống hệt nhau, DTW_norm phải gần 0. Với cùng từ nhưng tốc độ khác nhau, path có thể lệch đường chéo nhưng cost vẫn nhỏ hơn so với hai từ khác nhau. 

## **5.6. Bước 4 - Tạo templates và nhận dạng** 

<mark>from pathlib import Path LABELS = ['khong','mot','hai','ba','bon'] def build_templates(root): templates = {lab: [] for lab in LABELS} for lab in LABELS: files = sorted((Path(root)/lab).glob('*.wav'))[:3] for f in files: y = load_audio(f) y,_ = trim_energy(y) templates[lab].append(mfcc_feature(y)) return templates def recognize(path, templates): y = load_audio(path) y,_ = trim_energy(y) X = mfcc_feature(y) scores = {} for lab, refs in templates.items(): scores[lab] = min(dtw_distance(X, R)[0] for R in refs) pred = min(scores, key=scores.get) return pred, dict(sorted(scores.items(), key=lambda kv: kv[1]))</mark> 

## **5.7. Bước 5 - Đánh giá** 

<mark>from sklearn.metrics import confusion_matrix, accuracy_score</mark> 

<mark>y_true, y_pred = [], [] for lab in LABELS: test_files = sorted((Path('dataset')/lab).glob('*.wav'))[3:] for f in test_files: pred, scores = recognize(f, templates) y_true.append(lab); y_pred.append(pred) print(f.name, '->', pred, list(scores.items())[:3])</mark> 

Trang 12 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

<mark>print('Accuracy =', accuracy_score(y_true, y_pred)) print(confusion_matrix(y_true, y_pred, labels=LABELS))</mark> 

## **5.8. Kết quả cần quan sát trong case study** 

- DTW của hai lần nói cùng một từ thường nhỏ hơn DTW giữa hai từ khác nhau; không yêu cầu một ngưỡng số giống nhau giữa các nhóm vì micro/người nói/normalization khác nhau. 

- Optimal path của hai utterance cùng từ thường bám tương đối gần đường chéo nhưng có các đoạn ngang/dọc do kéo giãn hoặc nén thời gian. 

- Endpoint detection không tốt có thể làm DTW ưu tiên căn chỉnh silence, tăng chi phí và gây nhầm. 

- Các từ có cấu trúc âm học gần nhau hoặc bị phát âm rất ngắn có thể tạo ô ngoài đường chéo trong confusion matrix. 

- Cross-speaker thường khó hơn same-speaker vì MFCC vẫn chứa biến thiên liên quan người nói và kênh thu. 

## **5.9. Thí nghiệm mở rộng có kiểm soát** 

||**Mã**|**Thay đổi duy nhất**|**Giữ cố định**|**Câu hỏi cần trả lời**|
|---|---|---|---|---|
|E1||Không trim vs. có<br>trim|MFCC/DTW/templates|Endpoint ảnh hưởng<br>accuracy và path thế<br>nào?|
|E2||13 MFCC vs. 13<br>MFCC + Δ|Train/test split|Dynamic feature có<br>giảm nhầm không?|
|E3||1 template/từ vs. 3<br>templates/từ|Feature, test set|Nhiều reference<br>token giúp ổn định ra<br>sao?|
|E4||Same-speaker vs.<br>cross-speaker|Vocabulary/feature|Speaker variability<br>làm distance tăng bao<br>nhiêu?|



**6. Câu hỏi báo cáo** 

1. Vì sao không nên dùng toàn bộ waveform làm template chính khi hai utterance có thời lượng khác nhau? 

2. Giải thích vai trò khác nhau của short-time energy và ZCR trong endpoint detection. 

3. Vì sao Mel filterbank có khoảng cách theo Hz rộng dần khi tần số tăng? 

4. Log trong MFCC có tác dụng gì về mặt dynamic range? DCT biến M log-energy thành các hệ số gì? 

5. Trong ma trận DTW, ý nghĩa của bước ngang, bước dọc và bước chéo là gì? 

6. Tại sao phải chuẩn hóa DTW cost theo path length khi so sánh các utterance có thời lượng khác nhau? 

7. Nêu ít nhất ba nguyên nhân làm cùng một từ có MFCC khác nhau giữa hai lần nói. 

Trang 13 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

8. Từ confusion matrix, chọn cặp từ dễ nhầm nhất và phân tích waveform/MFCC/DTW path để đề xuất nguyên nhân. 

9. Nếu muốn hệ thống nhận dạng người nói mới chưa có template, DTW sẽ gặp hạn chế gì? Nội dung nào của Chương 3 sẽ giải quyết tốt hơn? 

## **7. Sản phẩm nộp** 

|**Tệp/sản phẩm**|**Nội dung tối thiểu**|
|---|---|
|Lab2_<MSSV>.ipynb|Code chạy từ đầu đến cuối, không phụ thuộc<br>biến đã chạy thủ công.|
|dataset/|Tập WAV đã dùng; nếu dung lượng lớn có<br>thể nộp link/thư mục dùng chung theo hướng<br>dẫn GV.|
|figures/|Waveform, energy/ZCR, MFCC heatmap,<br>DTW path, confusion matrix.|
|results.csv|Tên file test, nhãn thật, nhãn dự đoán, top-<br>1/top-2 DTW score.|
|Lab2_Report.pdf hoặc phần Markdown cuối<br>notebook|Bảng tham số, thí nghiệm E1–E2, trả lời câu<br>hỏi và kết luận.|



## **8. Tiêu chí chấm điểm (10 điểm)** 

|**Hạng mục**||**Điểm**|**Tiêu chí**|
|---|---|---|---|
|Dữ liệu + tiền xử lý|1.0||Đúng format, split hợp lý,<br>không data leakage.|
|Energy/ZCR + endpoint|1.5||Có công thức, đồ thị, trim<br>hoạt động và nhận xét.|
|MFCC|2.0||Thông số đúng, train/test<br>nhất quán, giải thích pipeline.|
|DTW tự cài đặt|2.5||Local distance, DP,<br>backtrack, normalization,<br>hình path.|
|Recognizer + đánh giá|1.5||Nhận dạng nhiều template,<br>accuracy, confusion matrix.|
|Thí nghiệm/phân tích|1.0||E1–E2 có đối chứng và nhận<br>xét dựa trên số liệu.|
|Trình bày/mã nguồn|0.5||Notebook sạch, chạy lại<br>được, biểu đồ có nhãn/đơn vị.|



## **9. Các lỗi thường gặp và cách kiểm tra** 

|**Lỗi**|**Dấu hiệu**|**Cách kiểm tra/sửa**|
|---|---|---|
|Không cùng sampling rate|MFCC shape/tần số khác bất<br>thường|Resample mọi file về 16 kHz<br>trước feature extraction.|



Trang 14 

**CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI     |     LAB 2** 

|**Lỗi**|**Dấu hiệu**|**Cách kiểm tra/sửa**|
|---|---|---|
|Data leakage|Accuracy gần 100% nhưng<br>vô nghĩa|Không dùng cùng utterance<br>làm template và test.|
|Endpoint cắt quá sát|Mất phụ âm đầu/cuối|Thêm margin và nghe lại file<br>trim.|
|MFCC bị transpose|DTW so sánh sai chiều|Mỗi hàng phải là một frame;<br>shape = (T,13).|
|Không chuẩn hóa DTW|File dài bị bất lợi|Chia cost cho path length.|
|Train/test khác pipeline|Distance tăng mạnh|Cùng pre-emphasis, window,<br>n_mels, n_mfcc, CMN.|
|Chỉ báo accuracy|Không biết nhầm ở đâu|Luôn nộp confusion matrix<br>và top-2 scores.|



**10. Tài liệu tham khảo** 

[1] Huang, X., Acero, A., & Hon, H-W. Spoken Language Processing. Prentice-Hall, 2001. Các phần liên quan: speech signal representations, MFCC, pattern recognition, DTW/HMM. 

[2] Rabiner, L.R., & Schafer, R.W. Theory and Applications of Digital Speech Processing. Pearson/Prentice Hall, 2011. Các phần liên quan: short-time energy/magnitude/ZCR, autocorrelation, MFCC/cepstral distance, endpoint detection. 

[3] Jurafsky, D., & Martin, J.H. Speech and Language Processing. Prentice-Hall, 2008. Nội dung nền tảng về speech recognition và mô hình hóa ngôn ngữ/nói. 

[4] Đề cương chi tiết học phần CSE457 – Xử lý âm thanh và tiếng nói, Trường Đại học Thủy lợi, 2023. 

**Ranh giới nội dung:** Lab 2 dừng ở template matching/DTW. HMM, mô hình âm học, mô hình ngôn ngữ và từ điển phát âm được dành cho Lab/Chương 3 theo đề cương CSE457. 

Trang 15 

