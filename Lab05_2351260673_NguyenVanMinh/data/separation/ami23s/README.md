# AMI EN2009d — dữ liệu ICA phần 3

Cùng đoạn **4941,96–4965,24 giây**, dài **23,28 giây**, mono 16 kHz PCM16 cho từng microphone.

| File | Vai trò |
|---|---|
| Array1-01.wav, Array1-05.wav | Hai microphone array thật, đầu vào X cho ICA |
| Headset-0.wav, Headset-1.wav | Mic gần người A/B, chỉ đối chiếu sau fit; không phải clean ground truth |
| source_A.wav, source_B.wav | Hai đoạn solo 3 giây từ bản mix của hội thoại, dùng tạo mixture mô phỏng |

Mic array và headset được lấy từ `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/EN2009d/audio/`, bằng HTTP Range đúng timestamp, không tải cả cuộc họp. Đây là bốn bản thu độc lập đồng bộ, không phải nhân đôi bản mono. Không dùng headset làm input rồi chấm điểm với chính headset đó.

Hai solo crops lấy từ bản preview phần 2, theo annotation thủ công: A ở 1,304–4,304s và B ở 17,768–20,768s. Chỉ cắt liên tục, không tổng hợp/ghép câu. Khi tạo mixture, hai đoạn khác thời điểm được chồng lên nhau có chủ đích: đây là mô phỏng, không phải overlap gốc.

`manifest.json` lưu URL nguồn, timestamp, role và SHA256 từng WAV. `intervals.json` giữ nhãn overlap/silence của preview để visualize, không được truyền vào ICA.

Nguồn: AMI Consortium / University of Edinburgh, AMI Meeting Corpus, EN2009d nonscenario. **CC BY 4.0**: https://groups.inf.ed.ac.uk/ami/corpus/license.shtml . Bản giấy phép đã lưu ở `data/speaker_recognition/ami/metadata/license.html`.

Tạo lại từ thư mục dự án: `.venv/bin/python tools/prepare_separation_data.py` (cần dữ liệu/metadata AMI phần 2 trước). Chạy `notebooks/Lab5_Part3_Separation.ipynb` bằng kernel `.venv/bin/python`.
