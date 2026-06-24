# Hệ thống dự đoán 3 càng đặc biệt xổ số miền Bắc

Mô hình dự đoán số xổ số sử dụng Recurrent Neural Network (RNN) với Long Short-Term Memory (LSTM) layers, được xây dựng bằng TensorFlow/Keras.

## Dự đoán ngày 25/06/2026

- **255 số đặc biệt:**
  - 467,370,661,974,634,298,794,188,997,426,177,059,309,725,135,138,639,903,981,965,830,744,808,496,503,821,985,963,095,427,620,685,152,286,148,407,492,092,155,463,533,619,768,898,509,547,716,047,287,713,325,761,448,883,748,099,697,860,125,971,623,791,596,699,792,433,027,909,116,208,461,525,766,535,022,405,911,442,512,826,410,364,278,877,647,604,756,066,379,682,628,707,412,406,521,282,698,866,204,130,705,931,919,170,333,352,240,645,191,185,432,884,662,520,555,723,211,197,371,493,230,106,495,912,004,478,956,915,993,044,563,571,088,302,413,052,946,777,908,077,578,835,819,372,275,254,190,465,180,746,683,854,863,019,196,862,430,524,617,206,102,457,110,930,237,291,957,354,968,570,375,914,307,889,486,967,668,818,123,879,479,470,977,464,686,198,528,028,690,045,327,294,880,016,715,789,648,338,343,758,476,674,787,153,836,043,015,109,638,657,687,727,453,026,346,897,741,652,953,929,118,058,165,872,605,192,262,646,616,119,234,838,553,320,403,613,599,861,905,341,484,729,515,315,042,081,501,799,910,772,363,272,076,085,824

## Kết quả dự đoán

| Ngày | 3 càng đặc biệt | 3 càng đầu |
|------|----------------|------------|
| **24/06/2026** | Số 751 - ❌ TRẬT | Số 602, 238, 063 - ❌ TRẬT |
| **23/06/2026** | Số 437 - ✅ TRÚNG | Số 240, 458, 180 - ✅ TRÚNG 1/3 |
| **22/06/2026** | Số 136 - ❌ TRẬT | Số 092, 906, 532 - ✅ TRÚNG 2/3 |
| **21/06/2026** | Số 083 - ✅ TRÚNG | Số 595, 625, 140 - ❌ TRẬT |
| **20/06/2026** | Số 860 - ✅ TRÚNG | Số 461, 038, 213 - ❌ TRẬT |
| **19/06/2026** | Số 155 - ❌ TRẬT | Số 021, 745, 601 - ❌ TRẬT |
| **18/06/2026** | Số 032 - ❌ TRẬT | Số 927, 271, 124 - ❌ TRẬT |
| **17/06/2026** | Số 203 - ❌ TRẬT | Số 411, 703, 695 - ❌ TRẬT |
| **16/06/2026** | Số 384 - ❌ TRẬT | Số 710, 036, 784 - ✅ TRÚNG 1/3 |
| **15/06/2026** | Số 680 - ❌ TRẬT | Số 774, 907, 243 - ✅ TRÚNG 2/3 |
| **14/06/2026** | Số 000 - ❌ TRẬT | Số 169, 931, 202 - ✅ TRÚNG 1/3 |
| **13/06/2026** | Số 014 - ❌ TRẬT | Số 752, 376, 448 - ✅ TRÚNG 1/3 |
| **12/06/2026** | Số 889 - ✅ TRÚNG | Số 063, 569, 896 - ✅ TRÚNG 1/3 |
| **11/06/2026** | Số 983 - ❌ TRẬT | Số 358, 179, 984 - ✅ TRÚNG 1/3 |
| **10/06/2026** | Số 308 - ✅ TRÚNG | Số 556, 218, 330 - ❌ TRẬT |
| **09/06/2026** | Số 251 - ✅ TRÚNG | Số 335, 887, 586 - ❌ TRẬT |
| **08/06/2026** | Số 505 - ✅ TRÚNG | Số 231, 092, 769 - ✅ TRÚNG 3/3 |
| **07/06/2026** | Số 393 - ❌ TRẬT | Số 968, 097, 715 - ❌ TRẬT |
| **06/06/2026** | Số 465 - ❌ TRẬT | Số 106, 675, 477 - ✅ TRÚNG 1/3 |
| **05/06/2026** | Số 978 - ❌ TRẬT | Số 945, 116, 522 - ✅ TRÚNG 1/3 |
| **04/06/2026** | Số 583 - ❌ TRẬT | Số 446, 045, 490 - ✅ TRÚNG 1/3 |
| **03/06/2026** | Số 636 - ❌ TRẬT | Số 374, 613, 055 - ✅ TRÚNG 1/3 |
| **02/06/2026** | Số 823 - ❌ TRẬT | Số 934, 425, 630 - ❌ TRẬT |
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
