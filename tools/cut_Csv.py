#!/usr/bin/env python3

def extract_first_n_lines(src_path, dst_path, n_lines=100000):
    try:
        with open(src_path, 'r', encoding='utf-8') as src_file, \
             open(dst_path, 'w', encoding='utf-8') as dst_file:
            
            for i, line in enumerate(src_file):
                if i >= n_lines:
                    break
                dst_file.write(line)
                
        print(f"Successfully wrote the first {n_lines} lines to {dst_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    source_file = "./data/filtered/filtered_aisdk-2025-01-23.csv"
    destination_file = "./data/small/1.csv"
    extract_first_n_lines(source_file, destination_file)
