#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tự động cập nhật phần dự đoán trong README.md
"""

import json
import os
import re
from datetime import datetime, timedelta

def read_latest_predictions():
    """Đọc dự đoán mới nhất từ data-predict.json"""
    if not os.path.exists("data-predict.json"):
        print("❌ Không tìm thấy file data-predict.json")
        return None, None
    
    try:
        with open("data-predict.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy dự đoán mới nhất
        if "predictions" in data and data["predictions"]:
            latest_prediction = data["predictions"][-1]  # Lấy bản ghi cuối cùng
            date = latest_prediction.get("date", "")
            numbers = latest_prediction.get("formatted_numbers", [])
            return date, numbers
        elif "date" in data:
            # Trường hợp file cũ chỉ có 1 bản ghi
            date = data.get("date", "")
            numbers = data.get("formatted_numbers", [])
            return date, numbers
        else:
            print("❌ Không tìm thấy dữ liệu dự đoán trong file")
            return None, None
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc file data-predict.json: {str(e)}")
        return None, None

def format_numbers_for_readme(numbers):
    """Format 255 số thành chuỗi dài cho README"""
    if not numbers or len(numbers) != 255:
        print(f"⚠️  Số lượng không đúng: {len(numbers) if numbers else 0}/255")
        return ""
    
    # Nối các số bằng dấu phẩy
    return ",".join(numbers)

def get_next_prediction_date():
    """Tính ngày dự đoán tiếp theo (ngày mai)"""
    tomorrow = datetime.now() + timedelta(days=1)
    return tomorrow.strftime("%d/%m/%Y")

def read_results_from_json():
    """Đọc kết quả dự đoán từ results.json"""
    if not os.path.exists("results.json"):
        print("⚠️  Không tìm thấy file results.json")
        return []
    
    try:
        with open("results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "results" in data and data["results"]:
            return data["results"]
        else:
            print("⚠️  Không có dữ liệu kết quả trong file")
            return []
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc file results.json: {str(e)}")
        return []

def read_prize6_results_from_json():
    """Đọc kết quả dự đoán giải 6 từ results-giai6.json"""
    if not os.path.exists("results-giai6.json"):
        print("⚠️  Không tìm thấy file results-giai6.json")
        return []
    
    try:
        with open("results-giai6.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "results" in data and data["results"]:
            return data["results"]
        else:
            print("⚠️  Không có dữ liệu kết quả giải 6 trong file")
            return []
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc file results-giai6.json: {str(e)}")
        return []

def format_results_for_readme(results, max_days=7):
    """Format kết quả thành chuỗi cho README"""
    if not results:
        return "- **3 càng đặc biệt:**\n  - Chưa có dữ liệu kết quả"
    
    # Sắp xếp theo ngày (mới nhất trước)
    sorted_results = sorted(results, key=lambda x: x.get("date", ""), reverse=True)
    
    # Lấy tối đa max_days kết quả
    recent_results = sorted_results[:max_days]
    
    results_text = "- **3 càng đặc biệt:**\n"
    
    for result in recent_results:
        date_str = result.get("date", "")
        special_number = result.get("special_number", "")
        status = result.get("status", "")
        
        # Chuyển đổi ngày từ YYYY-MM-DD sang DD/MM/YYYY
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%d/%m/%Y")
        except:
            formatted_date = date_str
        
        # Chuyển đổi status
        if status == "trúng":
            status_icon = "✅ TRÚNG"
        elif status == "trật":
            status_icon = "❌ TRẬT"
        else:
            status_icon = "❓ CHƯA RÕ"
        
        results_text += f"  - **{formatted_date}:** Số {special_number} - {status_icon}\n"
    
    return results_text.strip()

def format_prize6_results_for_readme(results, max_days=7):
    """Format kết quả giải 6 thành chuỗi cho README"""
    if not results:
        return "- **3 càng đầu:**\n  - Chưa có dữ liệu kết quả"
    
    # Sắp xếp theo ngày (mới nhất trước)
    sorted_results = sorted(results, key=lambda x: x.get("date", ""), reverse=True)
    
    # Lấy tối đa max_days kết quả
    recent_results = sorted_results[:max_days]
    
    results_text = "- **3 càng đầu:**\n"
    
    for result in recent_results:
        date_str = result.get("date", "")
        prize6_numbers = result.get("prize6_numbers", [])
        trung = result.get("trung", 0)
        trat = result.get("trat", 0)
        
        # Chuyển đổi ngày từ YYYY-MM-DD sang DD/MM/YYYY
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%d/%m/%Y")
        except:
            formatted_date = date_str
        
        # Tạo chuỗi hiển thị các số giải 6
        numbers_display = ", ".join(prize6_numbers) if prize6_numbers else "N/A"
        
        # Tạo status dựa trên số lượng trúng
        if trung > 0:
            status_icon = f"✅ TRÚNG {trung}/3"
        else:
            status_icon = "❌ TRẬT"
        
        results_text += f"  - **{formatted_date}:** Số {numbers_display} - {status_icon}\n"
    
    return results_text.strip()

def update_readme_section(date, numbers_str):
    """Cập nhật phần dự đoán và kết quả trong README.md"""
    if not os.path.exists("README.md"):
        print("❌ Không tìm thấy file README.md")
        return False
    
    try:
        # Đọc file README.md
        with open("README.md", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cập nhật phần dự đoán
        content = update_prediction_section(content, date, numbers_str)
        
        # Cập nhật phần kết quả
        content = update_results_section(content)
        
        # Ghi lại file
        with open("README.md", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Đã cập nhật file README.md thành công")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật README.md: {str(e)}")
        return False

def update_prediction_section(content, date, numbers_str):
    """Cập nhật phần dự đoán"""
    lines = content.split('\n')
    start_line = -1
    end_line = -1
    
    # Tìm dòng bắt đầu
    for i, line in enumerate(lines):
        if '## Dự đoán ngày' in line:
            start_line = i
            print(f"📝 Tìm thấy dòng bắt đầu dự đoán: {i+1}: {line}")
            break
    
    if start_line == -1:
        print("❌ Không tìm thấy phần dự đoán")
        return content
    
    # Tìm dòng kết thúc (dòng trống tiếp theo)
    for i in range(start_line + 1, len(lines)):
        if lines[i].strip() == '' and i > start_line + 3:  # Ít nhất 3 dòng sau header
            end_line = i
            print(f"📝 Tìm thấy dòng kết thúc dự đoán: {i+1}")
            break
    
    if end_line == -1:
        # Nếu không tìm thấy dòng trống, tìm dòng tiếp theo có ##
        for i in range(start_line + 1, len(lines)):
            if lines[i].startswith('##') and i > start_line + 2:
                end_line = i
                print(f"📝 Tìm thấy dòng kết thúc dự đoán (##): {i+1}")
                break
    
    if end_line == -1:
        print("❌ Không tìm thấy dòng kết thúc dự đoán")
        return content
    
    # Tạo nội dung mới cho phần dự đoán
    new_section = f"""## Dự đoán ngày {date}

- **255 số đặc biệt:**
  - {numbers_str}"""
    
    # Thay thế phần cũ
    new_lines = lines[:start_line] + new_section.split('\n') + lines[end_line:]
    print(f"✅ Đã cập nhật phần dự đoán cho ngày {date}")
    print(f"📊 Thay thế dự đoán từ dòng {start_line+1} đến {end_line}")
    
    return '\n'.join(new_lines)

def update_results_section(content):
    """Cập nhật phần kết quả"""
    # Đọc kết quả từ results.json
    results = read_results_from_json()
    results_text = format_results_for_readme(results)
    
    # Đọc kết quả giải 6 từ results-giai6.json
    prize6_results = read_prize6_results_from_json()
    prize6_text = format_prize6_results_for_readme(prize6_results)
    
    # Kết hợp cả hai phần kết quả
    combined_results = results_text + "\n\n" + prize6_text
    
    lines = content.split('\n')
    start_line = -1
    end_line = -1
    
    # Tìm dòng bắt đầu của phần kết quả
    for i, line in enumerate(lines):
        if '## Kết quả dự đoán' in line:
            # Tìm dòng có "- **3 càng đặc biệt:**" (bỏ qua dòng trống)
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and '- **3 càng đặc biệt:**' in lines[j]:
                    start_line = j
                    print(f"📝 Tìm thấy dòng bắt đầu kết quả: {i+1}: {line}")
                    print(f"📝 Dòng nội dung đầu tiên: {j+1}: '{lines[j]}'")
                    break
            break
    
    if start_line == -1:
        print("⚠️  Không tìm thấy phần kết quả, bỏ qua cập nhật")
        return content
    
    # Tìm dòng kết thúc (dòng mới bắt đầu với ##)
    for i in range(start_line + 1, len(lines)):
        if lines[i].startswith('##'):
            end_line = i
            print(f"📝 Tìm thấy dòng kết thúc kết quả: {i+1}")
            break
    
    if end_line == -1:
        end_line = len(lines)
    
    # Thay thế phần cũ
    new_lines = lines[:start_line] + combined_results.split('\n') + lines[end_line:]
    print(f"✅ Đã cập nhật phần kết quả (bao gồm giải 6)")
    print(f"📊 Thay thế kết quả từ dòng {start_line+1} đến {end_line}")
    
    return '\n'.join(new_lines)

def main():
    """Hàm chính"""
    print("=== CẬP NHẬT README.MD TỰ ĐỘNG ===\n")
    
    # Đọc dự đoán mới nhất
    date, numbers = read_latest_predictions()
    
    if not date or not numbers:
        print("❌ Không thể đọc dữ liệu dự đoán")
        return
    
    print(f"📅 Ngày dự đoán: {date}")
    print(f"🔢 Số lượng: {len(numbers)}")
    
    # Format số cho README
    numbers_str = format_numbers_for_readme(numbers)
    if not numbers_str:
        print("❌ Không thể format số")
        return
    
    # Chuyển đổi ngày từ YYYY-MM-DD sang DD/MM/YYYY
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d/%m/%Y")
    except:
        formatted_date = date  # Giữ nguyên nếu không parse được
    
    print(f"📝 Ngày format: {formatted_date}")
    print(f"📏 Độ dài chuỗi số: {len(numbers_str)} ký tự")
    
    # Hiển thị 10 số đầu và cuối để kiểm tra
    numbers_list = numbers_str.split(',')
    print(f"🔍 10 số đầu: {','.join(numbers_list[:10])}")
    print(f"🔍 10 số cuối: {','.join(numbers_list[-10:])}")
    
    # Hiển thị thông tin kết quả
    results = read_results_from_json()
    if results:
        display_count = min(7, len(results))
        print(f"\n📊 Kết quả dự đoán đặc biệt ({display_count}/{len(results)} ngày gần nhất):")
        for result in results[:3]:  # Hiển thị 3 ngày gần nhất
            date_str = result.get("date", "")
            special_number = result.get("special_number", "")
            status = result.get("status", "")
            
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                formatted_date_result = date_obj.strftime("%d/%m/%Y")
            except:
                formatted_date_result = date_str
            
            status_icon = "✅ TRÚNG" if status == "trúng" else "❌ TRẬT" if status == "trật" else "❓ CHƯA RÕ"
            print(f"  - {formatted_date_result}: Số {special_number} - {status_icon}")
    else:
        print("\n⚠️  Chưa có dữ liệu kết quả đặc biệt")
    
    # Hiển thị thông tin kết quả giải 6
    prize6_results = read_prize6_results_from_json()
    if prize6_results:
        display_count = min(7, len(prize6_results))
        print(f"\n📊 Kết quả dự đoán giải 6 ({display_count}/{len(prize6_results)} ngày gần nhất):")
        for result in prize6_results[:3]:  # Hiển thị 3 ngày gần nhất
            date_str = result.get("date", "")
            prize6_numbers = result.get("prize6_numbers", [])
            trung = result.get("trung", 0)
            
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                formatted_date_result = date_obj.strftime("%d/%m/%Y")
            except:
                formatted_date_result = date_str
            
            numbers_display = ", ".join(prize6_numbers) if prize6_numbers else "N/A"
            status_icon = f"✅ TRÚNG {trung}/3" if trung > 0 else "❌ TRẬT"
            print(f"  - {formatted_date_result}: Số {numbers_display} - {status_icon}")
    else:
        print("\n⚠️  Chưa có dữ liệu kết quả giải 6")
    
    # Cập nhật README
    if update_readme_section(formatted_date, numbers_str):
        print(f"\n{'='*60}")
        print("🎯 HOÀN THÀNH!")
        print(f"✅ Đã cập nhật README.md với dự đoán ngày {formatted_date}")
        print(f"✅ 255 số đặc biệt đã được cập nhật")
        print(f"✅ Phần kết quả đặc biệt đã được cập nhật từ results.json")
        print(f"✅ Phần kết quả giải 6 đã được cập nhật từ results-giai6.json")
        print(f"✅ Tổng cộng {len(numbers)} số dự đoán")
        print(f"{'='*60}")
    else:
        print("\n❌ Không thể cập nhật README.md")

if __name__ == "__main__":
    main()
