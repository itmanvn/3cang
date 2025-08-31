#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script debug để kiểm tra dữ liệu counts
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def analyze_counts_data():
    """Phân tích dữ liệu counts"""
    print("=== PHÂN TÍCH DỮ LIỆU COUNTS ===\n")
    
    # Đọc dữ liệu
    with open("data-dacbiet.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Lọc dữ liệu hợp lệ
    numbers = []
    for line in lines:
        line = line.strip()
        if line.isdigit() and len(line) == 3:
            numbers.append(int(line))
    
    print(f"Tổng số mẫu: {len(numbers)}")
    print(f"10 số đầu tiên: {numbers[:10]}")
    print(f"10 số cuối cùng: {numbers[-10:]}")
    
    # Phân tích chữ số
    print(f"\n=== PHÂN TÍCH CHỮ SỐ ===")
    all_digits = []
    for num in numbers:
        digits = [int(d) for d in str(num).zfill(3)]
        all_digits.extend(digits)
    
    # Thống kê chữ số
    digit_counts = [all_digits.count(i) for i in range(10)]
    print("Số lần xuất hiện của từng chữ số:")
    for i, count in enumerate(digit_counts):
        percentage = (count / len(all_digits)) * 100
        print(f"  Chữ số {i}: {count} lần ({percentage:.1f}%)")
    
    # Phân tích bộ đếm
    print(f"\n=== PHÂN TÍCH BỘ ĐẾM ===")
    digit_count_vectors = []
    for num in numbers:
        digits = [int(d) for d in str(num).zfill(3)]
        counts = [digits.count(i) for i in range(10)]
        digit_count_vectors.append(counts)
    
    digit_count_vectors = np.array(digit_count_vectors)
    print(f"Kích thước bộ đếm: {digit_count_vectors.shape}")
    print(f"Phạm vi giá trị: {digit_count_vectors.min()} - {digit_count_vectors.max()}")
    
    # Kiểm tra phân bố của từng chữ số
    print(f"\nPhân bố số lần xuất hiện của từng chữ số:")
    for i in range(10):
        unique, counts = np.unique(digit_count_vectors[:, i], return_counts=True)
        print(f"  Chữ số {i}: {dict(zip(unique, counts))}")
    
    # Tạo target cho mô hình
    print(f"\n=== PHÂN TÍCH TARGET ===")
    targets = np.argmax(digit_count_vectors, axis=1)
    unique_targets, target_counts = np.unique(targets, return_counts=True)
    
    print("Phân bố target (chữ số xuất hiện nhiều nhất):")
    for target, count in zip(unique_targets, target_counts):
        percentage = (count / len(targets)) * 100
        print(f"  Chữ số {target}: {count} lần ({percentage:.1f}%)")
    
    # Kiểm tra tính cân bằng
    print(f"\n=== KIỂM TRA TÍNH CÂN BẰNG ===")
    min_count = target_counts.min()
    max_count = target_counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Tỷ lệ mất cân bằng: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print("⚠️  Dữ liệu rất mất cân bằng!")
    elif imbalance_ratio > 5:
        print("⚠️  Dữ liệu mất cân bằng vừa phải")
    else:
        print("✅ Dữ liệu tương đối cân bằng")
    
    # Gợi ý cải thiện
    print(f"\n=== GỢI Ý CẢI THIỆN ===")
    if imbalance_ratio > 10:
        print("1. Sử dụng class weights trong mô hình")
        print("2. Áp dụng data augmentation")
        print("3. Sử dụng sampling techniques (SMOTE, etc.)")
        print("4. Giảm số epoch để tránh overfitting")
    
    return digit_count_vectors, targets

def test_scaler():
    """Kiểm tra scaler"""
    print(f"\n=== KIỂM TRA SCALER ===")
    
    # Tạo dữ liệu mẫu
    sample_data = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Chỉ có chữ số 0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Chỉ có chữ số 1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Chỉ có chữ số 2
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Có chữ số 0 và 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Chỉ có chữ số 9
    ])
    
    print(f"Dữ liệu mẫu:")
    print(sample_data)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sample_data)
    
    print(f"\nDữ liệu sau khi chuẩn hóa:")
    print(scaled_data)
    
    # Kiểm tra inverse transform
    original_data = scaler.inverse_transform(scaled_data)
    print(f"\nDữ liệu sau inverse transform:")
    print(original_data)
    
    return scaler

if __name__ == "__main__":
    # Phân tích dữ liệu
    digit_count_vectors, targets = analyze_counts_data()
    
    # Kiểm tra scaler
    scaler = test_scaler()
    
    print(f"\n=== KẾT LUẬN ===")
    print("Script debug hoàn thành. Kiểm tra output để xác định vấn đề.")
