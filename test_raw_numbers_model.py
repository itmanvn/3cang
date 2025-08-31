#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test mô hình raw_numbers cải tiến với 255 dự đoán
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import glob

def load_recent_data(data_file="data-dacbiet.txt", num_recent=10):
    """Đọc dữ liệu gần nhất từ file"""
    if not os.path.exists(data_file):
        print(f"Không tìm thấy file dữ liệu: {data_file}")
        return []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Lọc và chuyển đổi dữ liệu
    numbers = []
    for line in lines:
        line = line.strip()
        if line.isdigit() and len(line) == 3:
            numbers.append(int(line))
    
    # Trả về số gần nhất
    return numbers[-num_recent:]

def test_raw_numbers_model(model_path, scaler_path, recent_data):
    """Test mô hình raw_numbers với 255 dự đoán"""
    print(f"\n🔢 TESTING RAW_NUMBERS MODEL:")
    print(f"Model: {os.path.basename(model_path)}")
    
    try:
        # Load model và scaler
        model = tf.keras.models.load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
        
        # Chuẩn hóa dữ liệu đầu vào
        numbers_normalized = scaler.transform(np.array(recent_data).reshape(-1, 1)).flatten()
        
        # Dự đoán với randomness
        predictions = []
        current_sequence = numbers_normalized[-10:].reshape(1, 10, 1)
        
        print("🔄 Đang thực hiện dự đoán 255 số...")
        
        for i in range(255):
            if i % 50 == 0:  # Hiển thị tiến độ
                print(f"  Đã dự đoán {i}/255 số...")
            
            pred = model.predict(current_sequence, verbose=0)
            
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
            pred_original = int(scaler.inverse_transform([[pred_normalized]])[0][0])
            predictions.append(pred_original)
            
            # Cập nhật chuỗi
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_normalized
        
        print(f"✅ Dự đoán thành công: {len(predictions)} số")
        
        # Hiển thị 10 số đầu và 10 số cuối
        print(f"📊 10 số đầu tiên: {predictions[:10]}")
        print(f"📊 10 số cuối cùng: {predictions[-10:]}")
        
        # Kiểm tra tính đa dạng
        unique_predictions = len(set(predictions))
        print(f"📊 Số dự đoán khác nhau: {unique_predictions}/255")
        
        if unique_predictions >= 200:
            print("🎉 Tuyệt vời! Mô hình rất đa dạng!")
        elif unique_predictions >= 150:
            print("👍 Tốt! Mô hình đã đa dạng hơn!")
        elif unique_predictions >= 100:
            print("👍 Mô hình đã cải thiện!")
        else:
            print("⚠️  Mô hình vẫn còn lặp lại nhiều")
        
        # Lưu kết quả dự đoán vào file
        output_file = "predictions_255_numbers.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Dự đoán 255 số xổ số từ mô hình raw_numbers\n")
            f.write(f"# Model: {os.path.basename(model_path)}\n")
            f.write(f"# Dữ liệu gần nhất: {recent_data}\n")
            f.write(f"# Số dự đoán khác nhau: {unique_predictions}/255\n")
            f.write(f"# Thời gian: {os.popen('date').read().strip()}\n\n")
            
            for i, pred in enumerate(predictions, 1):
                f.write(f"{i:3d}: {pred:03d}\n")
        
        print(f"💾 Đã lưu kết quả vào file: {output_file}")
            
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")

def main():
    """Hàm chính"""
    print("=== TEST MÔ HÌNH RAW_NUMBERS CẢI TIẾN ===\n")
    
    # Tải dữ liệu gần nhất
    recent_data = load_recent_data()
    if not recent_data:
        print("Không thể đọc dữ liệu gần nhất")
        return
    
    print(f"📊 Dữ liệu gần nhất ({len(recent_data)} số): {recent_data}")
    
    # Tìm mô hình raw_numbers mới nhất
    raw_models = glob.glob("lottery_model_raw_numbers_*.keras")
    if not raw_models:
        print("❌ Không tìm thấy mô hình raw_numbers!")
        return
    
    # Sắp xếp theo thời gian sửa đổi
    raw_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_model = raw_models[0]
    
    print(f"\n🔍 Tìm thấy mô hình:")
    print(f"  raw_numbers: {os.path.basename(latest_model)}")
    
    # Test mô hình raw_numbers
    scaler_path = latest_model.replace('.keras', '_scaler.npy')
    if os.path.exists(scaler_path):
        test_raw_numbers_model(latest_model, scaler_path, recent_data)
    else:
        print(f"\n❌ Không tìm thấy scaler cho raw_numbers")
    
    print(f"\n{'='*60}")
    print("🎯 KẾT LUẬN:")
    print("✅ Mô hình raw_numbers đã được cải tiến với:")
    print("   - Regularization (L2, BatchNormalization)")
    print("   - Data augmentation (noise, rotation)")
    print("   - Temperature scaling (randomness)")
    print("   - Top-k sampling thay vì argmax")
    print("   - Kiến trúc tối ưu hóa")
    print("   - Dự đoán 255 số thay vì 5 số")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
