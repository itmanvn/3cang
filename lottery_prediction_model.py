#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô hình dự đoán số xổ số sử dụng RNN với LSTM layers
Hỗ trợ các loại dự đoán: raw_numbers, sum, counts
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class LotteryDataProcessor:
    """Xử lý dữ liệu xổ số"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Đọc dữ liệu từ file"""
        print("Đang đọc dữ liệu từ file...")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Lọc và chuyển đổi dữ liệu
        numbers = []
        for line in lines:
            line = line.strip()
            if line.isdigit() and len(line) == 3:
                numbers.append(int(line))
        
        print(f"Đã đọc {len(numbers)} số xổ số")
        return numbers
    
    def create_sequences(self, data, sequence_length=10):
        """Tạo chuỗi dữ liệu cho mô hình RNN"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_raw_numbers_data(self, sequence_length=10):
        """Chuẩn bị dữ liệu cho dự đoán số nguyên"""
        numbers = self.load_data()
        
        # Chuẩn hóa dữ liệu về khoảng [0, 1]
        numbers_array = np.array(numbers).reshape(-1, 1)
        numbers_normalized = self.scaler.fit_transform(numbers_array).flatten()
        
        # Tạo chuỗi
        X, y = self.create_sequences(numbers_normalized, sequence_length)
        
        # Chuyển đổi y về dạng one-hot encoding cho 1000 số (000-999)
        y_one_hot = tf.keras.utils.to_categorical(y * 999, num_classes=1000)
        
        return X, y_one_hot, self.scaler
    
    def prepare_sum_data(self, sequence_length=10):
        """Chuẩn bị dữ liệu cho dự đoán tổng các chữ số"""
        numbers = self.load_data()
        
        # Tính tổng các chữ số
        sums = []
        for num in numbers:
            digit_sum = sum(int(digit) for digit in str(num).zfill(3))
            sums.append(digit_sum)
        
        # Chuẩn hóa dữ liệu
        sums_array = np.array(sums).reshape(-1, 1)
        sums_normalized = self.scaler.fit_transform(sums_array).flatten()
        
        # Tạo chuỗi
        X, y = self.create_sequences(sums_normalized, sequence_length)
        
        # Chuyển đổi y về dạng one-hot encoding cho 28 số (0-27)
        y_one_hot = tf.keras.utils.to_categorical(y * 27, num_classes=28)
        
        return X, y_one_hot, self.scaler
    
    def prepare_counts_data(self, sequence_length=10):
        """Chuẩn bị dữ liệu cho dự đoán số lần xuất hiện của từng chữ số"""
        numbers = self.load_data()
        
        # Đếm số lần xuất hiện của từng chữ số (0-9)
        digit_counts = []
        for num in numbers:
            digits = [int(d) for d in str(num).zfill(3)]
            counts = [digits.count(i) for i in range(10)]
            digit_counts.append(counts)
        
        # Chuẩn hóa dữ liệu
        digit_counts_normalized = self.scaler.fit_transform(digit_counts)
        
        # Tạo chuỗi
        X, y = self.create_sequences(digit_counts_normalized, sequence_length)
        
        # Chuyển đổi y về dạng one-hot encoding cho 10 số (0-9)
        # Lấy chữ số xuất hiện nhiều nhất làm target (trước khi chuẩn hóa)
        y_digit = np.argmax(y, axis=1)
        y_one_hot = tf.keras.utils.to_categorical(y_digit, num_classes=10)
        
        print(f"Phân bố chữ số trong dữ liệu counts:")
        unique, counts = np.unique(y_digit, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"  Chữ số {digit}: {count} lần")
        
        return X, y_one_hot, self.scaler

class LotteryLSTMModel:
    """Mô hình LSTM cho dự đoán xổ số"""
    
    def __init__(self, input_shape, output_shape, model_type="raw_numbers"):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_type = model_type
        self.model = None
        self.history = None
        self.scaler = None  # Thêm thuộc tính scaler
        
    def build_model(self, lstm_units=128, dropout_rate=0.3):
        """Xây dựng mô hình LSTM"""
        if self.model_type == "counts":
            # Sử dụng kiến trúc đặc biệt cho counts với regularization mạnh hơn
            dropout_rate = 0.5  # Tăng dropout cho counts
            lstm_units = 64     # Giảm units để tránh overfitting
            
            model = keras.Sequential([
                # Input layer với noise
                layers.GaussianNoise(0.1, input_shape=self.input_shape),
                
                # LSTM layers với regularization mạnh
                layers.LSTM(lstm_units, return_sequences=True, 
                          kernel_regularizer=keras.regularizers.l2(0.01),
                          recurrent_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                layers.LSTM(lstm_units // 2, return_sequences=True,
                          kernel_regularizer=keras.regularizers.l2(0.01),
                          recurrent_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                layers.LSTM(lstm_units // 4,
                          kernel_regularizer=keras.regularizers.l2(0.01),
                          recurrent_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                # Dense layers với regularization
                layers.Dense(lstm_units // 2, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                layers.Dense(self.output_shape, activation='softmax')
            ])
        else:
            # Kiến trúc cải tiến cho raw_numbers và sum
            dropout_rate = 0.4  # Tăng dropout
            lstm_units = 96     # Giảm units để tránh overfitting
            
            model = keras.Sequential([
                # Input layer với noise nhẹ
                layers.GaussianNoise(0.05, input_shape=self.input_shape),
                
                # LSTM layers với regularization
                layers.LSTM(lstm_units, return_sequences=True, 
                          kernel_regularizer=keras.regularizers.l2(0.005),
                          recurrent_regularizer=keras.regularizers.l2(0.005)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                layers.LSTM(lstm_units // 2, return_sequences=True,
                          kernel_regularizer=keras.regularizers.l2(0.005),
                          recurrent_regularizer=keras.regularizers.l2(0.005)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                layers.LSTM(lstm_units // 4,
                          kernel_regularizer=keras.regularizers.l2(0.005),
                          recurrent_regularizer=keras.regularizers.l2(0.005)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                # Dense layers với regularization
                layers.Dense(lstm_units // 2, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.005)),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                
                layers.Dense(self.output_shape, activation='softmax')
            ])
        
        # Compile model với class weights nếu là counts
        if self.model_type == "counts":
            # Sử dụng optimizer và learning rate đặc biệt cho counts
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0005,  # Learning rate thấp hơn
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']  # Chỉ sử dụng accuracy cơ bản
            )
        else:
            # Sử dụng optimizer và learning rate đặc biệt cho raw_numbers và sum
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0008,  # Learning rate thấp hơn
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Huấn luyện mô hình"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Training với class weights nếu là counts
        if self.model_type == "counts":
            # Tính class weights thực tế từ dữ liệu
            from sklearn.utils.class_weight import compute_class_weight
            y_train_labels = np.argmax(y_train, axis=1)
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_labels),
                y=y_train_labels
            )
            class_weight_dict = dict(zip(range(10), class_weights))
            
            print(f"Class weights thực tế: {class_weight_dict}")
            
            # Data augmentation cho counts
            X_train_aug, y_train_aug = self._augment_counts_data(X_train, y_train)
            print(f"Dữ liệu sau augmentation: {X_train_aug.shape}")
            
            # Giảm epochs cho counts để tránh overfitting
            counts_epochs = min(epochs, 50)
            print(f"Sử dụng {counts_epochs} epochs cho counts")
            
            self.history = self.model.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_val, y_val),
                epochs=counts_epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight_dict,
                verbose=1
            )
        else:
            # Data augmentation nhẹ cho raw_numbers và sum
            X_train_aug, y_train_aug = self._augment_other_data(X_train, y_train)
            print(f"Dữ liệu sau augmentation: {X_train_aug.shape}")
            
            # Giảm epochs để tránh overfitting
            other_epochs = min(epochs, 80)
            print(f"Sử dụng {other_epochs} epochs cho {self.model_type}")
            
            self.history = self.model.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_val, y_val),
                epochs=other_epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        return self.history
    
    def _augment_counts_data(self, X, y):
        """Data augmentation cho dữ liệu counts"""
        if self.model_type != "counts":
            return X, y
        
        print("Đang thực hiện data augmentation cho counts...")
        
        # Tạo dữ liệu mới bằng cách thêm noise nhỏ
        X_aug = []
        y_aug = []
        
        # Thêm dữ liệu gốc
        X_aug.extend(X)
        y_aug.extend(y)
        
        # Tạo dữ liệu mới với noise
        for i in range(len(X)):
            # Thêm noise nhỏ vào input
            noise = np.random.normal(0, 0.01, X[i].shape)
            X_noisy = X[i] + noise
            X_noisy = np.clip(X_noisy, 0, 1)  # Giữ trong khoảng [0, 1]
            
            X_aug.append(X_noisy)
            y_aug.append(y[i])
        
        # Thêm dữ liệu với rotation nhỏ
        for i in range(len(X) // 2):  # Chỉ lấy một nửa để tránh quá nhiều
            # Xoay chuỗi một chút
            X_rotated = np.roll(X[i], shift=np.random.randint(-2, 3), axis=0)
            X_aug.append(X_rotated)
            y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)
    
    def _augment_other_data(self, X, y):
        """Data augmentation nhẹ cho raw_numbers và sum"""
        if self.model_type == "counts":
            return X, y
        
        print(f"Đang thực hiện data augmentation nhẹ cho {self.model_type}...")
        
        # Tạo dữ liệu mới bằng cách thêm noise nhỏ
        X_aug = []
        y_aug = []
        
        # Thêm dữ liệu gốc
        X_aug.extend(X)
        y_aug.extend(y)
        
        # Tạo dữ liệu mới với noise nhỏ
        for i in range(len(X)):
            # Thêm noise nhỏ vào input
            noise = np.random.normal(0, 0.005, X[i].shape)  # Noise nhỏ hơn counts
            X_noisy = X[i] + noise
            X_noisy = np.clip(X_noisy, 0, 1)  # Giữ trong khoảng [0, 1]
            
            X_aug.append(X_noisy)
            y_aug.append(y[i])
        
        # Thêm dữ liệu với rotation nhỏ
        for i in range(len(X) // 3):  # Chỉ lấy 1/3 để tránh quá nhiều
            # Xoay chuỗi một chút
            X_rotated = np.roll(X[i], shift=np.random.randint(-1, 2), axis=0)
            X_aug.append(X_rotated)
            y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)
    
    def predict(self, X):
        """Dự đoán"""
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Lưu mô hình"""
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        
        # Sử dụng định dạng .keras thay vì .h5 để tránh cảnh báo
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.keras')
        
        self.model.save(filepath)
        print(f"Đã lưu mô hình tại: {filepath}")
        
        # Lưu thêm scaler để sử dụng sau này (nếu có)
        if self.scaler is not None:
            scaler_path = filepath.replace('.keras', '_scaler.npy')
            np.save(scaler_path, self.scaler, allow_pickle=True)
            print(f"Đã lưu scaler tại: {scaler_path}")
        else:
            print("⚠️  Cảnh báo: Không có scaler để lưu")
    
    def plot_training_history(self):
        """Vẽ biểu đồ quá trình huấn luyện"""
        if self.history is None:
            print("Chưa có lịch sử huấn luyện")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

class LotteryPredictor:
    """Lớp dự đoán xổ số"""
    
    def __init__(self, model, scaler, model_type):
        self.model = model
        self.scaler = scaler
        self.model_type = model_type
    
    def predict_next_numbers(self, recent_numbers, num_predictions=5):
        """Dự đoán số tiếp theo"""
        if self.model_type == "raw_numbers":
            return self._predict_raw_numbers(recent_numbers, num_predictions)
        elif self.model_type == "sum":
            return self._predict_sum(recent_numbers, num_predictions)
        elif self.model_type == "counts":
            return self._predict_counts(recent_numbers, num_predictions)
        else:
            raise ValueError("Loại dự đoán không hợp lệ")
    
    def _predict_raw_numbers(self, recent_numbers, num_predictions):
        """Dự đoán số nguyên với randomness"""
        # Chuẩn hóa dữ liệu đầu vào
        numbers_normalized = self.scaler.transform(np.array(recent_numbers).reshape(-1, 1)).flatten()
        
        # Dự đoán với randomness
        predictions = []
        current_sequence = numbers_normalized[-10:].reshape(1, 10, 1)
        
        for _ in range(num_predictions):
            pred = self.model.predict(current_sequence, verbose=0)
            
            # Sử dụng temperature scaling để tăng randomness
            temperature = 1.5
            pred_scaled = pred[0] / temperature
            pred_probs = np.exp(pred_scaled) / np.sum(np.exp(pred_scaled))
            
            # Lấy top 5 predictions và chọn ngẫu nhiên
            top_5_indices = np.argsort(pred_probs)[-5:][::-1]
            top_5_probs = pred_probs[top_5_indices]
            
            # Chọn ngẫu nhiên từ top 5 với xác suất tương ứng
            chosen_idx = np.random.choice(top_5_indices, p=top_5_probs/np.sum(top_5_probs))
            pred_normalized = chosen_idx / 999.0
            
            # Chuyển về số nguyên
            pred_original = int(self.scaler.inverse_transform([[pred_normalized]])[0][0])
            predictions.append(pred_original)
            
            # Cập nhật chuỗi
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_normalized
        
        return predictions
    
    def _predict_sum(self, recent_numbers, num_predictions):
        """Dự đoán tổng các chữ số với randomness"""
        # Tính tổng các chữ số
        sums = []
        for num in recent_numbers:
            digit_sum = sum(int(digit) for digit in str(num).zfill(3))
            sums.append(digit_sum)
        
        # Chuẩn hóa dữ liệu
        sums_normalized = self.scaler.transform(np.array(sums).reshape(-1, 1)).flatten()
        
        # Dự đoán với randomness
        predictions = []
        current_sequence = sums_normalized[-10:].reshape(1, 10, 1)
        
        for _ in range(num_predictions):
            pred = self.model.predict(current_sequence, verbose=0)
            
            # Sử dụng temperature scaling để tăng randomness
            temperature = 1.5
            pred_scaled = pred[0] / temperature
            pred_probs = np.exp(pred_scaled) / np.sum(np.exp(pred_scaled))
            
            # Lấy top 5 predictions và chọn ngẫu nhiên
            top_5_indices = np.argsort(pred_probs)[-5:][::-1]
            top_5_probs = pred_probs[top_5_indices]
            
            # Chọn ngẫu nhiên từ top 5 với xác suất tương ứng
            chosen_idx = np.random.choice(top_5_indices, p=top_5_probs/np.sum(top_5_probs))
            pred_normalized = chosen_idx / 27.0
            
            # Chuyển về tổng gốc
            pred_original = int(self.scaler.inverse_transform([[pred_normalized]])[0][0])
            predictions.append(pred_original)
            
            # Cập nhật chuỗi
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_normalized
        
        return predictions
    
    def _predict_counts(self, recent_numbers, num_predictions):
        """Dự đoán chữ số xuất hiện nhiều nhất tiếp theo"""
        # Đếm số lần xuất hiện của từng chữ số
        digit_counts = []
        for num in recent_numbers:
            digits = [int(d) for d in str(num).zfill(3)]
            counts = [digits.count(i) for i in range(10)]
            digit_counts.append(counts)
        
        # Chuẩn hóa dữ liệu
        digit_counts_normalized = self.scaler.transform(digit_counts)
        
        # Dự đoán với randomness
        predictions = []
        current_sequence = digit_counts_normalized[-10:].reshape(1, 10, 10)
        
        for i in range(num_predictions):
            pred = self.model.predict(current_sequence, verbose=0)
            
            # Sử dụng temperature scaling để tăng randomness
            temperature = 2.0
            pred_scaled = pred[0] / temperature
            pred_probs = np.exp(pred_scaled) / np.sum(np.exp(pred_scaled))
            
            # Lấy top 3 predictions và chọn ngẫu nhiên
            top_3_indices = np.argsort(pred_probs)[-3:][::-1]
            top_3_probs = pred_probs[top_3_indices]
            
            # Chọn ngẫu nhiên từ top 3 với xác suất tương ứng
            chosen_idx = np.random.choice(top_3_indices, p=top_3_probs/np.sum(top_3_probs))
            predictions.append(chosen_idx)
            
            # Cập nhật chuỗi (sử dụng one-hot encoding)
            one_hot = np.zeros(10)
            one_hot[chosen_idx] = 1
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = one_hot
        
        return predictions

def main():
    """Hàm chính"""
    print("=== MÔ HÌNH DỰ ĐOÁN XỔ SỐ SỬ DỤNG RNN-LSTM ===\n")
    
    # Cấu hình
    DATA_FILE = "data-dacbiet.txt"
    SEQUENCE_LENGTH = 10
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Kiểm tra file dữ liệu
    if not os.path.exists(DATA_FILE):
        print(f"Không tìm thấy file dữ liệu: {DATA_FILE}")
        return
    
    # Xử lý dữ liệu
    processor = LotteryDataProcessor(DATA_FILE)
    
    # Danh sách các loại dự đoán - chỉ sử dụng raw_numbers
    prediction_types = ["raw_numbers"]
    
    for pred_type in prediction_types:
        print(f"\n{'='*50}")
        print(f"ĐANG XỬ LÝ LOẠI DỰ ĐOÁN: {pred_type.upper()}")
        print(f"{'='*50}")
        
        try:
            # Chuẩn bị dữ liệu
            if pred_type == "raw_numbers":
                X, y, scaler = processor.prepare_raw_numbers_data(SEQUENCE_LENGTH)
                output_shape = 1000
            elif pred_type == "sum":
                X, y, scaler = processor.prepare_sum_data(SEQUENCE_LENGTH)
                output_shape = 28
            elif pred_type == "counts":
                X, y, scaler = processor.prepare_counts_data(SEQUENCE_LENGTH)
                output_shape = 10
            
            print(f"Kích thước dữ liệu: X={X.shape}, y={y.shape}")
            
            # Chia dữ liệu
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Xây dựng mô hình
            if pred_type == "counts":
                input_features = X.shape[2]  # 10 features cho counts
            else:
                input_features = 1  # 1 feature cho raw_numbers và sum
            
            model_builder = LotteryLSTMModel(
                input_shape=(SEQUENCE_LENGTH, input_features),
                output_shape=output_shape,
                model_type=pred_type
            )
            
            # Lưu scaler vào model_builder
            model_builder.scaler = scaler
            
            # Huấn luyện mô hình
            print(f"\nBắt đầu huấn luyện mô hình {pred_type}...")
            history = model_builder.train(
                X_train, y_train, X_val, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE
            )
            
            # Đánh giá mô hình
            val_loss, val_accuracy = model_builder.model.evaluate(X_val, y_val, verbose=0)
            print(f"\nKết quả huấn luyện:")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Lưu mô hình
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"lottery_model_{pred_type}_{timestamp}.keras"
            model_builder.save_model(model_filename)
            
            # Vẽ biểu đồ
            model_builder.plot_training_history()
            
            # Dự đoán mẫu
            print(f"\nDự đoán mẫu cho {pred_type}:")
            predictor = LotteryPredictor(model_builder.model, scaler, pred_type)
            
            # Lấy 10 số gần nhất để dự đoán
            recent_data = processor.load_data()[-10:]
            
            if pred_type == "raw_numbers":
                predictions = predictor.predict_next_numbers(recent_data, 255)
                print(f"10 số gần nhất: {recent_data}")
                print(f"255 số dự đoán tiếp theo (hiển thị 10 số đầu): {predictions[:10]}...")
                print(f"Tổng cộng: {len(predictions)} số dự đoán")
            
        except Exception as e:
            print(f"Lỗi khi xử lý {pred_type}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("HOÀN THÀNH HUẤN LUYỆN TẤT CẢ MÔ HÌNH!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
