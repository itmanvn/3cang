# Hệ thống dự đoán 3 càng đặc biệt xổ số miền Bắc

Mô hình dự đoán số xổ số sử dụng Recurrent Neural Network (RNN) với Long Short-Term Memory (LSTM) layers, được xây dựng bằng TensorFlow/Keras.

## Dự đoán ngày 23/02/2026

- **255 số đặc biệt:**
  - 298,458,615,725,508,062,844,997,661,188,853,177,059,716,229,774,691,804,008,524,488,711,866,437,697,363,359,381,419,205,828,339,778,399,164,639,551,507,746,412,917,734,195,881,407,802,225,486,484,723,914,634,033,593,362,953,109,018,111,282,230,467,371,408,445,743,765,482,600,503,526,892,222,487,173,581,504,489,452,438,753,238,785,764,519,428,929,553,021,527,260,858,154,916,770,624,505,957,483,046,036,028,392,775,931,126,766,817,390,523,308,289,641,648,949,687,921,125,442,029,120,037,187,253,265,790,936,395,972,288,081,808,409,351,055,799,782,983,006,352,497,320,138,993,633,056,690,822,272,372,045,078,261,722,959,133,909,118,647,094,818,599,377,026,589,495,986,498,586,780,116,083,011,365,042,239,855,319,631,944,571,165,394,555,099,585,587,870,935,074,212,400,115,311,108,792,216,203,535,325,511,462,257,245,712,703,020,110,721,846,474,476,923,578,645,669,490,364,152,385,310,113,610,236,332,675,749,865,144,016,893,875,276,436,314,210,803,485,894,341,685,286,622,943,277,910,356,444,905,565,159,397,186,335,104

## Kết quả dự đoán

| Ngày | 3 càng đặc biệt | 3 càng đầu |
|------|----------------|------------|
| **22/02/2026** | Số 512 - ❌ TRẬT | Số 003, 214, 087 - ✅ TRÚNG 2/3 |
| **21/02/2026** | Số 413 - ❌ TRẬT | Số 243, 887, 936 - ✅ TRÚNG 1/3 |
| **20/02/2026** | Số 481 - ❌ TRẬT | Số 586, 074, 150 - ✅ TRÚNG 1/3 |
| **15/02/2026** | Số 601 - ❌ TRẬT | Số 634, 964, 259 - ✅ TRÚNG 1/3 |
| **14/02/2026** | Số 630 - ❌ TRẬT | Số 915, 781, 457 - ❌ TRẬT |
| **13/02/2026** | Số 177 - ✅ TRÚNG | Số 717, 473, 561 - ❌ TRẬT |
| **12/02/2026** | Số 475 - ❌ TRẬT | Số 990, 720, 207 - ✅ TRÚNG 1/3 |
| **11/02/2026** | Số 692 - ❌ TRẬT | Số 545, 709, 541 - ❌ TRẬT |
| **10/02/2026** | Số 390 - ❌ TRẬT | Số 632, 203, 423 - ❌ TRẬT |
| **09/02/2026** | Số 901 - ✅ TRÚNG | Số 130, 769, 434 - ✅ TRÚNG 1/3 |
| **08/02/2026** | Số 977 - ❌ TRẬT | Số 403, 432, 010 - ✅ TRÚNG 2/3 |
| **07/02/2026** | Số 385 - ✅ TRÚNG | Số 464, 234, 649 - ❌ TRẬT |
| **06/02/2026** | Số 133 - ✅ TRÚNG | Số 562, 720, 581 - ✅ TRÚNG 1/3 |
| **05/02/2026** | Số 268 - ❌ TRẬT | Số 664, 748, 487 - ❌ TRẬT |
| **04/02/2026** | Số 517 - ❌ TRẬT | Số 448, 371, 569 - ✅ TRÚNG 2/3 |
| **03/02/2026** | Số 648 - ❌ TRẬT | Số 552, 822, 147 - ✅ TRÚNG 2/3 |
| **02/02/2026** | Số 461 - ❌ TRẬT | Số 615, 883, 155 - ❌ TRẬT |
| **01/02/2026** | Số 377 - ❌ TRẬT | Số 431, 107, 573 - ✅ TRÚNG 1/3 |
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
