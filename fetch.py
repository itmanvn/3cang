#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script gọi fetch.py trong vietnam-lottery-xsmb-analysis/src và cập nhật data-dacbiet.txt
"""

import sys
import os
import subprocess
import json
from datetime import date

def run_fetch_script():
    """Chạy script fetch.py trong thư mục vietnam-lottery-xsmb-analysis/src"""
    print("🔄 Đang chạy script fetch.py trong vietnam-lottery-xsmb-analysis/src...")
    
    try:
        # Thay đổi thư mục làm việc
        original_cwd = os.getcwd()
        os.chdir('vietnam-lottery-xsmb-analysis')
        
        # Chạy script fetch.py từ thư mục gốc để đường dẫn data/ đúng
        result = subprocess.run([sys.executable, 'src/fetch.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        # Quay lại thư mục gốc
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print("✅ Script fetch.py đã chạy thành công")
            print("📊 Output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("❌ Script fetch.py chạy thất bại")
            print("📋 Lỗi:")
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy script fetch.py: {str(e)}")
        # Quay lại thư mục gốc nếu có lỗi
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return False

def read_xsmb_data():
    """Đọc dữ liệu từ file xsmb.json"""
    print("📚 Đang đọc dữ liệu từ xsmb.json...")
    
    try:
        xsmb_file = "vietnam-lottery-xsmb-analysis/data/xsmb.json"
        
        if not os.path.exists(xsmb_file):
            print(f"❌ Không tìm thấy file: {xsmb_file}")
            return None
        
        with open(xsmb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Đã đọc {len(data)} bản ghi từ xsmb.json")
        return data
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc file xsmb.json: {str(e)}")
        return None

def extract_special_numbers(xsmb_data):
    """Trích xuất 3 số cuối của giải đặc biệt từ dữ liệu xsmb"""
    print("🔍 Đang trích xuất 3 số cuối của giải đặc biệt...")
    
    try:
        special_numbers = []
        
        for record in xsmb_data:
            if 'special' in record and record['special']:
                special_number = record['special']
                last_3_digits = special_number % 1000
                special_numbers.append(last_3_digits)
        
        print(f"✅ Đã trích xuất {len(special_numbers)} số từ giải đặc biệt")
        
        # Hiển thị 5 số gần nhất
        if special_numbers:
            print(f"📊 5 số gần nhất: {special_numbers[-5:]}")
        
        return special_numbers
        
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất số: {str(e)}")
        return None

def update_data_dacbiet(special_numbers):
    """Cập nhật file data-dacbiet.txt với tất cả số từ xsmb.json (xóa trắng trước)"""
    print("💾 Đang cập nhật file data-dacbiet.txt...")
    
    try:
        data_file = "data-dacbiet.txt"
        
        # Kiểm tra file hiện tại
        existing_lines = 0
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                existing_lines = len(f.readlines())
            print(f"📚 File hiện tại có {existing_lines} dòng")
        
        # Xóa trắng file trước khi cập nhật
        print(f"🧹 Đang xóa trắng file {data_file}...")
        with open(data_file, 'w', encoding='utf-8') as f:
            pass  # Tạo file trống
        
        # Ghi tất cả số mới vào file (cho phép trùng lặp)
        print(f"🆕 Đang ghi {len(special_numbers)} số mới vào file...")
        
        with open(data_file, 'w', encoding='utf-8') as f:
            for number in special_numbers:
                f.write(f"{number:03d}\n")
        
        print(f"✅ Đã ghi {len(special_numbers)} số mới vào file {data_file}")
        
        # Hiển thị thống kê
        print(f"📊 Tổng số dòng trong file: {len(special_numbers)}")
        print(f"📊 Dữ liệu cũ đã được xóa, chỉ còn dữ liệu mới")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật file: {str(e)}")
        return False



def main():
    """Hàm chính"""
    print("=== GỌI FETCH.PY VÀ CẬP NHẬT DATA-DACBIET.TXT ===\n")
    
    # Bước 1: Chạy script fetch.py trong vietnam-lottery-xsmb-analysis/src
    print("🔄 BƯỚC 1: Chạy script fetch.py...")
    if not run_fetch_script():
        print("❌ Không thể chạy script fetch.py")
        return
    
    print()
    
    # Bước 2: Đọc dữ liệu từ xsmb.json
    print("🔄 BƯỚC 2: Đọc dữ liệu từ xsmb.json...")
    xsmb_data = read_xsmb_data()
    if xsmb_data is None:
        print("❌ Không thể đọc dữ liệu từ xsmb.json")
        return
    
    print()
    
    # Bước 3: Trích xuất 3 số cuối của giải đặc biệt
    print("🔄 BƯỚC 3: Trích xuất 3 số cuối của giải đặc biệt...")
    special_numbers = extract_special_numbers(xsmb_data)
    if special_numbers is None:
        print("❌ Không thể trích xuất số từ dữ liệu")
        return
    
    print()
    
    # Bước 4: Cập nhật file data-dacbiet.txt
    print("🔄 BƯỚC 4: Cập nhật file data-dacbiet.txt...")
    success = update_data_dacbiet(special_numbers)
    
    if success:
        print(f"\n{'='*60}")
        print("🎯 HOÀN THÀNH!")
        print(f"✅ Đã chạy script fetch.py thành công")
        print(f"✅ Đã đọc {len(xsmb_data)} bản ghi từ xsmb.json")
        print(f"✅ Đã trích xuất {len(special_numbers)} số từ giải đặc biệt")
        print(f"✅ Đã cập nhật file data-dacbiet.txt")
        print(f"{'='*60}")
    else:
        print(f"\n⚠️  Cập nhật không thành công")

if __name__ == "__main__":
    main()
