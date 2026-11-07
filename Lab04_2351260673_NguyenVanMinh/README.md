# Lab 04 — Text-to-Speech

Nguyễn Văn Minh — 2351260673. Báo cáo ngắn: [REPORT.md](REPORT.md). Yêu cầu gốc: [SPEC.md](SPEC.md).

## Chạy lại

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/download_model.py
python -m src.train_hmm
bash scripts/run_pipeline.sh
bash scripts/benchmark.sh
pytest -q
```

Chỉ bước cài thư viện và tải model cần Internet. `download_model.py` kiểm tra SHA-256 được lưu trong manifest; trọng số Piper không đưa vào Git. HMM đã huấn luyện được lưu trong `data/models/hmm.json`. Pipeline dùng cả ba backend và xuất âm thanh, hình và bảng đo cho ba câu. Mô hình Piper/VITS tiếng Việt `vi_VN-vais1000-medium`, bản `v1.0.0`: [model card](https://huggingface.co/rhasspy/piper-voices/blob/v1.0.0/vi/vi_VN/vais1000/medium/MODEL_CARD). Piper sử dụng GPL-3.0; model card ghi bộ dữ liệu theo CC BY 4.0.

## Nghe và đánh giá

Mở `evaluation/listening.html` bằng trình duyệt. Chỉ đưa người nghe file này cùng các audio `S*.wav`; giữ `organizer_key.csv` riêng. Mỗi người nghe nhập mã khác nhau, chép lại nội dung nghe được và chấm naturalness/continuity 1–5, rồi tải CSV. Người tổ chức đối chiếu từ đúng với câu chuẩn trong REPORT, ánh xạ sample sang method bằng bảng khóa và nhập ratings theo `ratings_template.csv`. Không cung cấp câu chuẩn và tên phương pháp cho người nghe trước khi chấm. Chạy hàm `summarize_ratings` để tổng hợp điểm thực.

Phần mềm và đầu ra đã có; **kết quả đánh giá chủ quan còn chờ người nghe thật**. Không có MOS/intelligibility giả lập. HMM là mô hình nhỏ theo word-unit, có hạn chế về phát âm và F0 ở vùng vô thanh; đây là kết quả thực nghiệm, chưa phải hệ thống TTS sản xuất.
