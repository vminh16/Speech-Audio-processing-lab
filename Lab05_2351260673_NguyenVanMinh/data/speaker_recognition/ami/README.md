# AMI EN2009d — tập con hội thoại tự nhiên

Nguồn: AMI Meeting Corpus, The AMI Consortium / University of Edinburgh.

- Meeting: **EN2009d**, loại **nonscenario** theo metadata chính thức (họp thực, không phải đọc kịch bản hay đọc số).
- Audio: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/HeadsetAudio/EN2009d.Mix-Headset.wav
- Annotation thủ công v1.6.2: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
- Giấy phép hiện hành: **CC BY 4.0**, https://groups.inf.ed.ac.uk/ami/corpus/license.shtml ; bản trang giấy phép được lưu trong `metadata/license.html`.
- Ghi công: Carletta et al., The AMI Meeting Corpus: A Pre-announcement (2005).

## Người nói và phân chia

| Nhãn lab | AMI agent | Global speaker | Enrollment | Dev | Test |
|---|---|---|---:|---:|---:|
| spk01 | A | FEE083 | 4 | 2 | 2 |
| spk02 | B | FEE096 | 4 | 2 | 2 |
| spk03 | C | MEE094 | 4 | 2 | 2 |

Mỗi người lấy 8 lượt nói có segment ID khác nhau, phân bố xuyên cuộc họp theo thứ tự thời gian. Bốn lượt đầu enrollment, hai lượt tiếp dev, hai lượt cuối test. Các clip không chồng thời gian, không dùng lại cùng lượt nói ở nhiều split. Dùng cùng một bản mix cho mọi người để tránh khác biệt microphone riêng thành dấu hiệu nhận dạng nhãn.

## Cách cắt

Chọn đoạn liên tục 2–4 giây tại ranh giới từ, tối thiểu 5 từ; từ được annotate phủ ít nhất 60% độ dài đoạn. Loại đoạn có bất kỳ speaker khác (kể cả agent D) nói chồng theo annotation, thêm biên 100 ms. Không ghép, pad, resample, denoise hoặc normalize; nguồn vốn mono 16 kHz PCM16. Giữ biên độ gốc. Cắt có thể lấy một phần của lượt nói dài, không bảo đảm là câu hoàn chỉnh.

`conversation_23s.wav` là đoạn liên tục 4941,96–4965,24 giây từ nguồn, dài 23,28 giây. Toàn khoảng preview được loại khỏi tập chọn speaker clips. Đây là audio hội thoại để nghe trước, không có nhãn tự động dùng làm ground truth.

`manifest.json` ghi file, speaker, segment ID, start/end trong bản nguồn, transcript thủ công tương ứng và SHA256. `metadata/` giữ segments/words cho đủ 4 speaker để có thể kiểm tra overlap, cùng meeting.xml và thông tin provenance.

## Giới hạn

Đây là audio headset mix thực tế, vẫn có âm phòng và có thể có crosstalk/âm không được annotate; không xem là clean reference tuyệt đối cho enhancement. Các split nằm trong cùng cuộc họp, đủ làm baseline nhận dạng và verification của lab nhưng không phải đánh giá khác phiên/khác thiết bị. Phương pháp MFCC/cosine được triển khai trong `src/speaker.py`; chạy tại `notebooks/Lab5_Part2_Speaker.ipynb`.

Tạo lại: `python3 tools/prepare_ami.py` từ thư mục dự án. Có thể dùng archive annotation đã tải: `python3 tools/prepare_ami.py --annotations /path/to/ami_public_manual_1.6.2.zip`. Chỉ tải HTTP Range các đoạn cần thiết; nếu server không hỗ trợ, script dừng thay vì tải cả file dài.
