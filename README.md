# Mô hình dự đoán xổ số sử dụng RNN-LSTM

Hệ thống dự đoán số xổ số sử dụng Recurrent Neural Network (RNN) với Long Short-Term Memory (LSTM) layers, được xây dựng bằng TensorFlow/Keras.

## Tính năng

- **3 loại dự đoán:**
  - `raw_numbers`: Dự đoán số xổ số nguyên (000-999)
  - `sum`: Dự đoán tổng các chữ số (0-27)
  - `counts`: Dự đoán số lần xuất hiện của từng chữ số (0-9)

- **Kiến trúc mô hình:**
  - 3 LSTM layers với dropout
  - Dense layers với activation functions
  - Early stopping và learning rate reduction
  - Validation split 20%

## Cài đặt

1. **Cài đặt Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Kiểm tra file dữ liệu:**
   - Đảm bảo file `data-dacbiet.txt` chứa dữ liệu xổ số (mỗi dòng một số 3 chữ số)

## Sử dụng

### 1. Huấn luyện mô hình

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

### 2. Kiểm tra mô hình đã huấn luyện

Kiểm tra trạng thái các mô hình:

```bash
python check_models.py
```

Script sẽ:
- Hiển thị danh sách tất cả mô hình
- Kiểm tra trạng thái và kích thước
- Xác minh scaler tương ứng
- Kiểm tra file dữ liệu

### 3. Dự đoán sử dụng mô hình đã huấn luyện

Sau khi huấn luyện xong, sử dụng script dự đoán:

```bash
python predict_lottery.py
```

Script sẽ:
- Tự động tìm mô hình mới nhất
- Đọc dữ liệu gần nhất
- Thực hiện dự đoán
- Hiển thị kết quả chi tiết

### 4. Test mô hình raw_numbers

Test mô hình với 255 dự đoán:

```bash
python test_raw_numbers_model.py
```

Script sẽ:
- Tải mô hình raw_numbers mới nhất
- Thực hiện dự đoán 255 số
- Kiểm tra tính đa dạng của dự đoán
- Lưu kết quả vào file `predictions_255_numbers.txt`

## Cấu trúc file

```
├── lottery_prediction_model.py    # Script huấn luyện chính (chỉ raw_numbers)
├── predict_lottery.py             # Script dự đoán (255 số)
├── check_models.py                # Script kiểm tra mô hình
├── test_raw_numbers_model.py      # Script test mô hình raw_numbers
├── requirements.txt               # Dependencies
├── data-dacbiet.txt              # Dữ liệu xổ số
├── README.md                     # Hướng dẫn này
├── lottery_model_raw_numbers_*.keras  # Mô hình raw_numbers (định dạng mới)
├── lottery_model_raw_numbers_*_scaler.npy  # Scaler tương ứng
```

## Cấu hình mô hình

### Tham số có thể điều chỉnh:

- `SEQUENCE_LENGTH`: Độ dài chuỗi đầu vào (mặc định: 10)
- `EPOCHS`: Số epoch huấn luyện (mặc định: 100)
- `BATCH_SIZE`: Kích thước batch (mặc định: 32)
- `lstm_units`: Số units trong LSTM layers (mặc định: 128)
- `dropout_rate`: Tỷ lệ dropout (mặc định: 0.3)

### Kiến trúc mô hình:

```
Input (sequence_length, features)
    ↓
LSTM(128, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
LSTM(64, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
LSTM(32)
    ↓
Dropout(0.3)
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(output_shape, activation='softmax')
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

### 4. Mô hình counts dự đoán lặp lại
Nếu mô hình counts chỉ dự đoán một chữ số:
- Chạy `python debug_counts.py` để phân tích dữ liệu
- Huấn luyện lại với regularization mạnh hơn
- Sử dụng data augmentation và temperature scaling

## Hiệu suất

- **Thời gian huấn luyện:** Khoảng 10-30 phút tùy thuộc vào phần cứng
- **Độ chính xác:** Thường đạt 15-25% trên validation set
- **Kích thước mô hình:** Khoảng 2-5 MB mỗi file .keras

## Đóng góp

Để cải thiện mô hình, có thể:
- Thêm features mới (ngày tháng, mùa, v.v.)
- Thử nghiệm kiến trúc khác (GRU, Transformer)
- Sử dụng ensemble methods
- Tối ưu hóa hyperparameters

## License

Dự án này chỉ dành cho mục đích nghiên cứu và giáo dục.
