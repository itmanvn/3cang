# Hệ thống dự đoán 3 càng đặc biệt xổ số miền Bắc

Mô hình dự đoán số xổ số sử dụng Recurrent Neural Network (RNN) với Long Short-Term Memory (LSTM) layers, được xây dựng bằng TensorFlow/Keras.

## Dự đoán ngày 27/09/2025

- **255 số đặc biệt:**
  - 132,131,617,115,508,177,879,405,858,448,661,362,681,188,615,554,458,777,716,083,978,424,237,598,479,913,795,157,103,666,651,971,273,398,156,977,668,622,974,058,193,874,546,014,580,219,798,107,394,869,794,299,082,454,729,934,949,451,296,675,841,161,062,526,003,656,513,985,924,307,545,981,813,648,220,078,031,246,329,209,540,361,126,755,486,303,109,660,993,714,009,018,863,614,152,165,411,939,960,138,024,687,525,175,898,711,907,054,827,857,270,099,120,330,779,552,445,950,752,975,623,970,499,856,581,032,335,529,685,169,536,484,820,673,428,945,629,020,560,154,948,826,980,198,725,143,408,076,351,231,562,658,995,455,806,518,179,715,087,796,713,747,663,401,792,676,903,883,520,044,440,698,370,145,801,967,097,693,125,464,025,567,211,456,884,576,172,537,354,710,830,283,816,144,478,127,644,771,395,895,095,410,055,940,234,982,988,015,000,279,891,232,213,387,067,040,357,255,066,174,192,383,558,338,538,200,487,501,065,768,151,064,563,053,390,833,199,441,417,818,733,430,659,313,201,236,002,088,130,262,909,468,459,527,906

## Kết quả dự đoán

| Ngày | 3 càng đặc biệt | 3 càng đầu |
|------|----------------|------------|
| **26/09/2025** | Số 223 - ❌ TRẬT | Số 287, 186, 349 - ❌ TRẬT |
| **25/09/2025** | Số 263 - ❌ TRẬT | Số 424, 875, 042 - ❌ TRẬT |
| **24/09/2025** | Số 859 - ❌ TRẬT | Số 284, 831, 408 - ❌ TRẬT |
| **23/09/2025** | Số 888 - ❌ TRẬT | Số 509, 775, 640 - ✅ TRÚNG 1/3 |
| **22/09/2025** | Số 461 - ❌ TRẬT | Số 604, 201, 895 - ✅ TRÚNG 2/3 |
| **21/09/2025** | Số 471 - ❌ TRẬT | Số 697, 783, 794 - ❌ TRẬT |
| **20/09/2025** | Số 590 - ✅ TRÚNG | Số 455, 691, 286 - ✅ TRÚNG 1/3 |
| **19/09/2025** | Số 846 - ❌ TRẬT | Số 528, 353, 362 - ✅ TRÚNG 1/3 |
| **18/09/2025** | Số 450 - ❌ TRẬT | Số 475, 989, 682 - ❌ TRẬT |
| **17/09/2025** | Số 005 - ✅ TRÚNG | Số 634, 523, 318 - ❌ TRẬT |
| **16/09/2025** | Số 705 - ❌ TRẬT | Số 479, 389, 851 - ❌ TRẬT |
| **15/09/2025** | Số 946 - ❌ TRẬT | Số 224, 616, 465 - ✅ TRÚNG 2/3 |
| **14/09/2025** | Số 807 - ❌ TRẬT | Số 825, 466, 649 - ✅ TRÚNG 1/3 |
| **13/09/2025** | Số 401 - ❌ TRẬT | Số 257, 334, 253 - ✅ TRÚNG 1/3 |
| **12/09/2025** | Số 686 - ❌ TRẬT | Số 926, 348, 349 - ✅ TRÚNG 2/3 |
| **11/09/2025** | Số 217 - ❌ TRẬT | Số 350, 643, 296 - ✅ TRÚNG 1/3 |
| **10/09/2025** | Số 231 - ❌ TRẬT | Số 062, 205, 325 - ❌ TRẬT |
| **09/09/2025** | Số 460 - ✅ TRÚNG | Số 388, 087, 085 - ❌ TRẬT |
| **08/09/2025** | Số 493 - ❌ TRẬT | Số 814, 134, 074 - ✅ TRÚNG 2/3 |
| **07/09/2025** | Số 137 - ❌ TRẬT | Số 055, 957, 432 - ❌ TRẬT |
| **06/09/2025** | Số 093 - ✅ TRÚNG | Số 473, 753, 153 - ❌ TRẬT |
| **05/09/2025** | Số 878 - ❌ TRẬT | Số 370, 237, 517 - ❌ TRẬT |
| **04/09/2025** | Số 943 - ❌ TRẬT | Số 913, 668, 770 - ✅ TRÚNG 1/3 |
| **03/09/2025** | Số 033 - ✅ TRÚNG | Số 272, 145, 363 - ❌ TRẬT |
| **02/09/2025** | Số 079 - ❌ TRẬT | Số 423, 682, 886 - ✅ TRÚNG 1/3 |
| **01/09/2025** | Số 335 - ❌ TRẬT | Số 846, 167, 326 - ❌ TRẬT |
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
