# Tổng hợp kết quả Lab 3 (sinh tự động bởi `scripts/report.py`)

## So sánh DTW và Gaussian HMM

| Chỉ số | DTW (Lab 2) | Gaussian HMM (Lab 3) |
| --- | --- | --- |
| Accuracy (%) | 100.0 | 100.0 |
| Số mẫu train / lớp | 6 | 6 |
| Thời gian train (ms) | 0.01 | 362.5 |
| Thời gian test / trial (ms) | 49.10 | 1.13 |
| Cặp từ dễ nhầm nhất | không có | không có |
| Cơ chế alignment | Warping path | Viterbi state path |

## Bảng log-likelihood (mỗi lớp một file test)

| file | label | ll_mo | ll_dong | ll_len | ll_xuong | ll_dung | pred | margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mo_07.wav | mo | -11769.2 | -12494.4 | -13766.3 | -12653.0 | -12805.6 | mo | 725.2 |
| dong_07.wav | dong | -9000.1 | -8356.3 | -8922.3 | -8942.5 | -8879.7 | dong | 523.4 |
| len_07.wav | len | -8781.2 | -8401.1 | -7950.6 | -8741.9 | -8461.0 | len | 450.5 |
| xuong_07.wav | xuong | -11609.0 | -11418.1 | -12280.5 | -10775.5 | -11246.7 | xuong | 471.2 |
| dung_07.wav | dung | -14980.4 | -14577.5 | -15465.3 | -15099.6 | -13964.5 | dung | 613.0 |

## E1 - Số trạng thái

| n_states | train_acc | test_acc | converged | mean_iters | train_ll | test_ll | train_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1.000 | 1.000 | 5/5 | 15.400 | -52.509 | -53.517 | 0.120 |
| 5 | 1.000 | 1.000 | 5/5 | 28.400 | -50.626 | -52.557 | 0.349 |
| 7 | 1.000 | 1.000 | 5/5 | 34.600 | -49.732 | -52.358 | 0.532 |

## E2 - Chiều đặc trưng

| features | train_acc | test_acc | converged | mean_iters | train_ll | test_ll | train_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13-D | 1.000 | 1.000 | 5/5 | 15.600 | -15.382 | -16.432 | 0.145 |
| 39-D | 1.000 | 1.000 | 5/5 | 28.400 | -50.626 | -52.557 | 0.342 |

## E3 - Kiểu covariance

| cov_type | train_acc | test_acc | converged | mean_iters | train_ll | test_ll | train_time_s | ll_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diag | 1.000 | 1.000 | 5/5 | 28.400 | -50.626 | -52.557 | 0.346 | 1.931 |
| full | 1.000 | 0.600 | 5/5 | 19.200 | -41.120 | -60.039 | 3.367 | 18.919 |

## E4 - Số mẫu train / lớp

| n_train | method | mean_acc | std_acc | runs |
| --- | --- | --- | --- | --- |
| 1 | HMM | 0.600 | 0.126 | 5 |
| 1 | DTW | 0.640 | 0.196 | 5 |
| 2 | HMM | 0.880 | 0.098 | 5 |
| 2 | DTW | 0.840 | 0.080 | 5 |
| 3 | HMM | 1.000 | 0.000 | 5 |
| 3 | DTW | 0.800 | 0.126 | 5 |
| 4 | HMM | 1.000 | 0.000 | 5 |
| 4 | DTW | 1.000 | 0.000 | 5 |
| 6 | HMM | 1.000 | 0.000 | 1 |
| 6 | DTW | 1.000 | 0.000 | 1 |

## E5 - Độ bền vững với nhiễu

| condition | method | mean_acc | std_acc | runs |
| --- | --- | --- | --- | --- |
| clean | HMM | 1.000 | 0.000 | 1 |
| clean | DTW | 1.000 | 0.000 | 1 |
| SNR 20 dB | HMM | 0.800 | 0.000 | 3 |
| SNR 20 dB | DTW | 0.800 | 0.000 | 3 |
| SNR 10 dB | HMM | 0.600 | 0.000 | 3 |
| SNR 10 dB | DTW | 0.600 | 0.000 | 3 |
| SNR 5 dB | HMM | 0.533 | 0.094 | 3 |
| SNR 5 dB | DTW | 0.600 | 0.000 | 3 |
| SNR 0 dB | HMM | 0.467 | 0.094 | 3 |
| SNR 0 dB | DTW | 0.467 | 0.094 | 3 |

## E6 - Trọng số LM

| condition | lambda_lm | accuracy | trials |
| --- | --- | --- | --- |
| clean | 0.000 | 1.000 | 15 |
| clean | 0.100 | 1.000 | 15 |
| clean | 0.300 | 1.000 | 15 |
| clean | 1.000 | 1.000 | 15 |
| SNR 10 dB | 0.000 | 0.600 | 45 |
| SNR 10 dB | 0.100 | 0.600 | 45 |
| SNR 10 dB | 0.300 | 0.600 | 45 |
| SNR 10 dB | 1.000 | 0.667 | 45 |
| SNR 5 dB | 0.000 | 0.556 | 45 |
| SNR 5 dB | 0.100 | 0.533 | 45 |
| SNR 5 dB | 0.300 | 0.489 | 45 |
| SNR 5 dB | 1.000 | 0.556 | 45 |
| SNR 0 dB | 0.000 | 0.444 | 45 |
| SNR 0 dB | 0.100 | 0.533 | 45 |
| SNR 0 dB | 0.300 | 0.489 | 45 |
| SNR 0 dB | 1.000 | 0.489 | 45 |

## Bảng P(cur | prev) của bigram LM

| prev | mo | dong | len | xuong | dung |
| --- | --- | --- | --- | --- | --- |
| &lt;s&gt; | 0.294 | 0.294 | 0.176 | 0.176 | 0.059 |
| mo | 0.059 | 0.059 | 0.412 | 0.294 | 0.176 |
| dong | 0.059 | 0.059 | 0.294 | 0.412 | 0.176 |
| len | 0.059 | 0.059 | 0.059 | 0.059 | 0.765 |
| xuong | 0.059 | 0.059 | 0.059 | 0.059 | 0.765 |
| dung | 0.412 | 0.412 | 0.059 | 0.059 | 0.059 |

Tên hiển thị: `mo` = mở, `dong` = đóng, `len` = lên, `xuong` = xuống, `dung` = dừng
