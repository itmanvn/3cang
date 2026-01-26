# Hệ thống dự đoán 3 càng đặc biệt xổ số miền Bắc

Mô hình dự đoán số xổ số sử dụng Recurrent Neural Network (RNN) với Long Short-Term Memory (LSTM) layers, được xây dựng bằng TensorFlow/Keras.

## Dự đoán ngày 27/01/2026

- **255 số đặc biệt:**
  - 661,173,639,362,725,199,448,997,971,426,122,739,458,665,188,059,736,298,409,218,033,705,634,490,385,496,140,299,305,593,974,763,392,582,494,955,352,394,286,166,438,144,427,615,569,507,663,324,330,466,981,422,693,953,841,222,112,376,605,303,349,538,128,526,447,792,677,062,380,749,069,227,081,043,047,147,189,421,301,656,604,309,378,958,453,778,142,288,344,580,638,512,910,878,789,480,810,246,644,050,061,216,113,249,233,840,814,027,174,614,969,724,770,225,138,257,356,425,211,239,820,934,446,161,439,341,933,238,440,828,114,899,575,248,880,368,688,730,926,377,587,581,497,975,405,369,437,126,511,564,898,220,735,408,285,777,003,524,451,063,395,436,540,935,965,414,034,895,625,732,835,570,169,722,505,914,431,867,054,606,916,851,389,202,717,328,548,295,347,741,462,989,875,073,685,563,647,100,401,635,801,561,534,106,825,498,481,423,988,149,397,666,632,800,500,040,543,264,737,339,332,088,819,014,729,774,331,277,470,279,245,589,630,696,412,263,715,527,176,896,780,117,460,968,012,155,224,404,020,435,983,845,450,430,556

## Kết quả dự đoán

| Ngày | 3 càng đặc biệt | 3 càng đầu |
|------|----------------|------------|
| **26/01/2026** | Số 974 - ❌ TRẬT | Số 294, 739, 215 - ✅ TRÚNG 1/3 |
| **25/01/2026** | Số 230 - ❌ TRẬT | Số 489, 940, 371 - ✅ TRÚNG 2/3 |
| **24/01/2026** | Số 062 - ❌ TRẬT | Số 811, 013, 869 - ✅ TRÚNG 1/3 |
| **23/01/2026** | Số 022 - ❌ TRẬT | Số 484, 417, 202 - ✅ TRÚNG 1/3 |
| **22/01/2026** | Số 063 - ✅ TRÚNG | Số 819, 774, 727 - ✅ TRÚNG 1/3 |
| **21/01/2026** | Số 186 - ❌ TRẬT | Số 480, 226, 435 - ❌ TRẬT |
| **20/01/2026** | Số 878 - ✅ TRÚNG | Số 916, 392, 879 - ✅ TRÚNG 1/3 |
| **19/01/2026** | Số 286 - ❌ TRẬT | Số 674, 109, 851 - ❌ TRẬT |
| **18/01/2026** | Số 151 - ❌ TRẬT | Số 139, 283, 310 - ❌ TRẬT |
| **17/01/2026** | Số 824 - ❌ TRẬT | Số 800, 491, 957 - ✅ TRÚNG 1/3 |
| **16/01/2026** | Số 128 - ❌ TRẬT | Số 132, 666, 595 - ❌ TRẬT |
| **15/01/2026** | Số 522 - ❌ TRẬT | Số 497, 368, 374 - ❌ TRẬT |
| **14/01/2026** | Số 817 - ❌ TRẬT | Số 945, 187, 978 - ✅ TRÚNG 1/3 |
| **13/01/2026** | Số 027 - ❌ TRẬT | Số 508, 652, 762 - ✅ TRÚNG 1/3 |
| **12/01/2026** | Số 894 - ❌ TRẬT | Số 301, 697, 335 - ✅ TRÚNG 2/3 |
| **11/01/2026** | Số 438 - ❌ TRẬT | Số 768, 195, 519 - ✅ TRÚNG 1/3 |
| **10/01/2026** | Số 793 - ❌ TRẬT | Số 762, 712, 486 - ❌ TRẬT |
| **09/01/2026** | Số 523 - ❌ TRẬT | Số 417, 102, 243 - ❌ TRẬT |
| **08/01/2026** | Số 162 - ❌ TRẬT | Số 021, 854, 337 - ❌ TRẬT |
| **07/01/2026** | Số 389 - ✅ TRÚNG | Số 065, 764, 088 - ✅ TRÚNG 1/3 |
| **06/01/2026** | Số 447 - ❌ TRẬT | Số 354, 932, 356 - ❌ TRẬT |
| **05/01/2026** | Số 505 - ❌ TRẬT | Số 525, 408, 194 - ✅ TRÚNG 1/3 |
| **04/01/2026** | Số 397 - ❌ TRẬT | Số 239, 577, 634 - ✅ TRÚNG 1/3 |
| **03/01/2026** | Số 949 - ❌ TRẬT | Số 512, 176, 433 - ✅ TRÚNG 1/3 |
| **02/01/2026** | Số 748 - ❌ TRẬT | Số 641, 853, 159 - ❌ TRẬT |
| **01/01/2026** | Số 068 - ❌ TRẬT | Số 731, 617, 253 - ❌ TRẬT |
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
