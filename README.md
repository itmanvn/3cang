# Hệ thống dự đoán 3 càng đặc biệt xổ số miền Bắc

Mô hình dự đoán số xổ số sử dụng Recurrent Neural Network (RNN) với Long Short-Term Memory (LSTM) layers, được xây dựng bằng TensorFlow/Keras.

## Dự đoán ngày 27/12/2025

- **255 số đặc biệt:**
  - 127,597,661,298,794,370,917,508,188,965,356,526,242,274,062,019,271,725,708,362,640,903,199,219,448,132,033,997,844,964,439,101,335,368,191,418,156,973,540,574,472,802,969,849,171,913,166,782,288,570,265,975,865,226,748,046,118,626,561,702,970,073,198,140,403,654,427,361,227,250,697,718,107,647,030,423,891,021,209,303,615,360,117,530,025,430,438,203,249,042,932,283,176,558,517,566,510,211,052,928,551,478,149,925,190,897,535,282,674,129,029,842,971,175,011,456,347,054,390,012,899,004,147,946,060,553,138,924,987,568,090,816,678,100,822,125,169,316,629,538,930,744,956,015,800,527,671,780,722,881,938,153,923,480,222,767,384,094,330,603,065,624,174,454,236,633,808,102,307,353,716,137,036,280,318,731,179,460,728,648,253,461,992,078,729,374,264,974,918,248,587,059,916,063,331,567,218,170,097,983,327,556,893,506,798,935,315,732,902,977,910,089,433,743,693,756,285,963,545,233,486,352,685,909,437,675,659,328,801,445,262,929,766,583,876,588,772,182,455,901,419,906,741,807,165,779,828,960,745,825,229,962,824,862,610

## Kết quả dự đoán

| Ngày | 3 càng đặc biệt | 3 càng đầu |
|------|----------------|------------|
| **26/12/2025** | Số 636 - ❌ TRẬT | Số 798, 369, 784 - ❌ TRẬT |
| **25/12/2025** | Số 199 - ❌ TRẬT | Số 049, 203, 442 - ✅ TRÚNG 2/3 |
| **24/12/2025** | Số 629 - ❌ TRẬT | Số 752, 692, 446 - ✅ TRÚNG 1/3 |
| **23/12/2025** | Số 059 - ❌ TRẬT | Số 724, 117, 034 - ✅ TRÚNG 2/3 |
| **22/12/2025** | Số 496 - ✅ TRÚNG | Số 492, 608, 506 - ✅ TRÚNG 1/3 |
| **21/12/2025** | Số 036 - ✅ TRÚNG | Số 514, 178, 302 - ❌ TRẬT |
| **20/12/2025** | Số 816 - ✅ TRÚNG | Số 133, 543, 763 - ✅ TRÚNG 1/3 |
| **19/12/2025** | Số 372 - ❌ TRẬT | Số 021, 276, 825 - ✅ TRÚNG 1/3 |
| **18/12/2025** | Số 138 - ✅ TRÚNG | Số 775, 279, 459 - ❌ TRẬT |
| **17/12/2025** | Số 484 - ❌ TRẬT | Số 005, 086, 859 - ❌ TRẬT |
| **16/12/2025** | Số 792 - ✅ TRÚNG | Số 509, 291, 871 - ✅ TRÚNG 1/3 |
| **15/12/2025** | Số 093 - ❌ TRẬT | Số 597, 253, 521 - ❌ TRẬT |
| **14/12/2025** | Số 968 - ❌ TRẬT | Số 962, 614, 898 - ❌ TRẬT |
| **13/12/2025** | Số 753 - ❌ TRẬT | Số 230, 946, 852 - ✅ TRÚNG 1/3 |
| **12/12/2025** | Số 836 - ❌ TRẬT | Số 095, 715, 254 - ❌ TRẬT |
| **11/12/2025** | Số 668 - ✅ TRÚNG | Số 996, 689, 242 - ❌ TRẬT |
| **10/12/2025** | Số 052 - ❌ TRẬT | Số 264, 231, 162 - ❌ TRẬT |
| **09/12/2025** | Số 908 - ❌ TRẬT | Số 974, 352, 054 - ❌ TRẬT |
| **08/12/2025** | Số 192 - ❌ TRẬT | Số 447, 034, 136 - ❌ TRẬT |
| **07/12/2025** | Số 626 - ✅ TRÚNG | Số 377, 095, 490 - ✅ TRÚNG 2/3 |
| **06/12/2025** | Số 526 - ❌ TRẬT | Số 311, 191, 630 - ❌ TRẬT |
| **05/12/2025** | Số 235 - ❌ TRẬT | Số 135, 872, 032 - ✅ TRÚNG 1/3 |
| **04/12/2025** | Số 729 - ❌ TRẬT | Số 770, 428, 511 - ❌ TRẬT |
| **03/12/2025** | Số 038 - ✅ TRÚNG | Số 454, 637, 093 - ❌ TRẬT |
| **02/12/2025** | Số 986 - ✅ TRÚNG | Số 611, 124, 619 - ❌ TRẬT |
| **01/12/2025** | Số 312 - ❌ TRẬT | Số 481, 960, 637 - ✅ TRÚNG 2/3 |
## Tính năng

- **Mô hình chính:**
  - `raw_numbers`: Dự đoán số xổ số nguyên (000-999) với 255 dự đoán khác nhau

- **Kiến trúc mô hình:**
  - 3 LSTM layers với dropout và regularization
  - GaussianNoise, BatchNormalization, L2 regularization
  - Data augmentation và temperature scaling
  - Early stopping và learning rate reduction
  - Validation split 20%

## Cài đặt

### 1. Clone repository và submodule

```bash
# Clone repository chính
git clone https://github.com/itmanvn/3cang.git
cd 3cang

# Clone và cập nhật submodule vietnam-lottery-xsmb-analysis
git submodule update --init --recursive
```

**Lưu ý quan trọng:** Repository này sử dụng git submodule để quản lý dự án `vietnam-lottery-xsmb-analysis`. Đảm bảo bạn đã clone đầy đủ submodule để script `fetch.py` hoạt động chính xác.

### 2. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
cd vietnam-lottery-xsmb-analysis
pip install -r requirements.txt
```

### 3. Kiểm tra file dữ liệu
   - Đảm bảo file `data-dacbiet.txt` chứa dữ liệu xổ số (mỗi dòng một số 3 chữ số)
   - Đảm bảo thư mục `vietnam-lottery-xsmb-analysis` đã được clone đầy đủ

## Sử dụng

### 1. Lấy kết quả xổ số và cập nhật dữ liệu

Lấy kết quả xổ số mới nhất và cập nhật vào file dữ liệu:

```bash
python fetch.py
```

Script sẽ:
- **Bước 1:** Chạy script `fetch.py` trong `vietnam-lottery-xsmb-analysis/src` để fetch dữ liệu mới
- **Bước 2:** Đọc dữ liệu từ `vietnam-lottery-xsmb-analysis/data/xsmb.json` (đã được cập nhật)
- **Bước 3:** Trích xuất 3 số cuối của giải đặc biệt từ tất cả bản ghi
- **Bước 4:** Cập nhật file `data-dacbiet.txt` với các số mới (nếu có)
- **Kết quả:** Đồng bộ dữ liệu giữa `xsmb.json` và `data-dacbiet.txt`

### 2. Huấn luyện mô hình

Chạy script chính để huấn luyện mô hình raw_numbers:

```bash
python lottery_prediction_model.py
```

Script sẽ:
- Đọc dữ liệu từ `data-dacbiet.txt`
- Chuẩn bị dữ liệu cho dự đoán raw_numbers
- Huấn luyện mô hình LSTM
- Lưu mô hình dưới dạng file `.keras` (định dạng mới)
- Lưu scaler tương ứng để sử dụng dự đoán
- Thực hiện dự đoán 255 số mẫu

### 3. Kiểm tra mô hình đã huấn luyện

Kiểm tra trạng thái các mô hình:

```bash
python check_models.py
```

Script sẽ:
- Hiển thị danh sách tất cả mô hình
- Kiểm tra trạng thái và kích thước
- Xác minh scaler tương ứng
- Kiểm tra file dữ liệu

### 4. Dự đoán sử dụng mô hình đã huấn luyện

Sau khi huấn luyện xong, sử dụng script dự đoán:

```bash
python predict_lottery.py
```

Script sẽ:
- Tự động tìm mô hình mới nhất
- Đọc dữ liệu gần nhất
- Thực hiện dự đoán
- Hiển thị kết quả chi tiết

### 5. Dự đoán 255 số khác nhau từ mô hình

Dự đoán 255 số khác nhau hoàn toàn từ mô hình raw_numbers:

```bash
python predict_255_unique_from_model.py
```

Script sẽ:
- Tải mô hình raw_numbers mới nhất
- Thực hiện dự đoán 255 số khác nhau hoàn toàn
- Sử dụng temperature scaling cao (3.0) và top-10 sampling
- Lưu kết quả vào file `data-predict.json` với định dạng JSON

## Cấu trúc repository

```
3cang/
├── lottery_prediction_model.py    # Script huấn luyện chính (chỉ raw_numbers)
├── predict_lottery.py             # Script dự đoán cơ bản
├── predict_255_unique_from_model.py  # Script dự đoán 255 số khác nhau
├── update_readme.py               # Script cập nhật README.md tự động
├── fetch.py                      # Script lấy kết quả xổ số và cập nhật dữ liệu
├── check_models.py                # Script kiểm tra mô hình
├── cleanup_models.py              # Script dọn dẹp model cũ
├── requirements.txt               # Dependencies
├── data-dacbiet.txt              # Dữ liệu xổ số
├── data-predict.json             # Kết quả dự đoán 255 số (JSON)
├── results.json                  # Kết quả kiểm tra dự đoán
├── README.md                     # Hướng dẫn này
├── lottery_model_raw_numbers_*.keras  # Mô hình raw_numbers (định dạng mới)
├── lottery_model_raw_numbers_*_scaler.npy  # Scaler tương ứng
├── .gitmodules                   # Cấu hình git submodule
└── vietnam-lottery-xsmb-analysis/  # Git submodule (dữ liệu xổ số)
    ├── src/
    │   ├── lottery.py            # Module xử lý dữ liệu xổ số
    │   └── fetch.py              # Script fetch dữ liệu từ web
    ├── data/
    │   └── xsmb.json             # Dữ liệu xổ số gốc
    └── README.md                 # Hướng dẫn submodule
```

**Lưu ý:** Thư mục `vietnam-lottery-xsmb-analysis` là một git submodule chứa dữ liệu xổ số và các script xử lý dữ liệu. Script `fetch.py` trong thư mục gốc sẽ gọi script trong submodule này để cập nhật dữ liệu.

## Cấu hình mô hình

### Tham số có thể điều chỉnh:

- `SEQUENCE_LENGTH`: Độ dài chuỗi đầu vào (mặc định: 10)
- `EPOCHS`: Số epoch huấn luyện (mặc định: 100, giảm xuống 80 cho raw_numbers)
- `BATCH_SIZE`: Kích thước batch (mặc định: 32)
- `lstm_units`: Số units trong LSTM layers (mặc định: 96 cho raw_numbers)
- `dropout_rate`: Tỷ lệ dropout (mặc định: 0.4 cho raw_numbers)
- `temperature`: Temperature scaling cho dự đoán (mặc định: 3.0)
- `top_k`: Số predictions top-k cho sampling (mặc định: 10)

### Kiến trúc mô hình raw_numbers:

```
Input (10, 1) - Chuỗi 10 số gần nhất
    ↓
GaussianNoise(0.05) - Thêm noise nhẹ
    ↓
LSTM(96, return_sequences=True) + L2 regularization
    ↓
Dropout(0.4) + BatchNormalization
    ↓
LSTM(48, return_sequences=True) + L2 regularization
    ↓
Dropout(0.4) + BatchNormalization
    ↓
LSTM(24) + L2 regularization
    ↓
Dropout(0.4) + BatchNormalization
    ↓
Dense(48, activation='relu') + L2 regularization
    ↓
Dropout(0.4) + BatchNormalization
    ↓
Dense(1000, activation='softmax') - 1000 số từ 000-999
```

## Loại dự đoán

### 1. Raw Numbers (Số nguyên) - **MÔ HÌNH CHÍNH**
- **Input:** Chuỗi 10 số xổ số gần nhất
- **Output:** Dự đoán 255 số tiếp theo (000-999)
- **Độ chính xác:** Phụ thuộc vào tính ngẫu nhiên của xổ số
- **Đặc biệt:** Sử dụng regularization, data augmentation và temperature scaling để tăng đa dạng dự đoán
- **Ứng dụng:** Dự đoán số xổ số với độ đa dạng cao

## Lưu ý quan trọng

⚠️ **Cảnh báo:** 
- Đây chỉ là mô hình AI dựa trên dữ liệu lịch sử
- **KHÔNG đảm bảo** kết quả dự đoán chính xác
- Xổ số là trò chơi may rủi, không thể dự đoán chính xác 100%
- Chỉ sử dụng cho mục đích nghiên cứu và giải trí

## Xử lý lỗi thường gặp

### 1. Lỗi CUDA/GPU
Nếu gặp lỗi GPU, có thể chuyển sang CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### 2. Lỗi memory
Giảm `batch_size` hoặc `sequence_length` nếu gặp lỗi out of memory.

### 3. Lỗi dữ liệu
Đảm bảo file `data-dacbiet.txt` chứa đúng định dạng:
- Mỗi dòng một số
- Số có đúng 3 chữ số
- Không có ký tự đặc biệt

### 4. Mô hình raw_numbers dự đoán lặp lại
Nếu mô hình raw_numbers dự đoán lặp lại nhiều:
- Sử dụng `python predict_255_unique_from_model.py` để đảm bảo 255 số khác nhau
- Tăng temperature scaling (từ 1.5 lên 3.0)
- Tăng top-k sampling (từ top-5 lên top-10)
- Sử dụng data augmentation và regularization mạnh hơn

## Hiệu suất

- **Thời gian huấn luyện:** Khoảng 5-15 phút cho mô hình raw_numbers
- **Độ chính xác:** Thường đạt 15-25% trên validation set
- **Kích thước mô hình:** Khoảng 1.5-2 MB cho file .keras
- **Thời gian dự đoán:** Khoảng 1-2 phút cho 255 số khác nhau
- **Tính đa dạng:** Đảm bảo 255 số khác nhau hoàn toàn (100%)

## Quản lý Git Submodule

### Cập nhật submodule

```bash
# Cập nhật submodule lên phiên bản mới nhất
git submodule update --remote

# Hoặc cập nhật submodule cụ thể
cd vietnam-lottery-xsmb-analysis
git pull origin main
cd ..
git add vietnam-lottery-xsmb-analysis
git commit -m "Update vietnam-lottery-xsmb-analysis submodule"
```

### Clone repository với submodule

```bash
# Clone với submodule (khuyến nghị)
git clone --recurse-submodules https://github.com/your-username/3cang.git

# Hoặc clone riêng lẻ
git clone https://github.com/your-username/3cang.git
cd 3cang
git submodule init
git submodule update
```

### Kiểm tra trạng thái submodule

```bash
# Xem trạng thái submodule
git submodule status

# Xem thông tin chi tiết
git submodule foreach git status
```

### Xử lý lỗi submodule

Nếu gặp lỗi với submodule:
```bash
# Xóa và clone lại submodule
rm -rf vietnam-lottery-xsmb-analysis
git submodule update --init --recursive

# Hoặc reset submodule về trạng thái commit
git submodule update --force --recursive
```

## Đóng góp

Để cải thiện mô hình, có thể:
- Thêm features mới (ngày tháng, mùa, v.v.)
- Thử nghiệm kiến trúc khác (GRU, Transformer)
- Sử dụng ensemble methods
- Tối ưu hóa hyperparameters
- Cải thiện temperature scaling và top-k sampling
- Thêm data augmentation techniques mới
- Tham khảo thêm: https://www.beatlottery.co.uk/lottery-predictions

## License

Dự án này chỉ dành cho mục đích nghiên cứu và giáo dục.
