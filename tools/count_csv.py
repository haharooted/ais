import os

file_path = "./data/rawcsv/aisdk-2025-01-27.csv"

def count_lines_smart(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

row_count = count_lines_smart(file_path)
print(f"Total Rows (including header): {row_count}")
print(f"Total Data Rows (excluding header): {row_count - 1 if row_count > 0 else 0}")
