import pyarrow.csv as pv
import pyarrow.compute as pc
import pyarrow as pa
import os
import argparse
import glob

def search_csv_folder(folder_path, column_name, search_value):
    all_results = []

    for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            convert_options = pv.ConvertOptions(
                column_types={column_name: pa.string()}  # Force string conversion
            )
            read_options = pv.ReadOptions(use_threads=True)

            table = pv.read_csv(file_path, read_options=read_options, convert_options=convert_options)

            if column_name not in table.column_names:
                print(f"Warning: Column '{column_name}' not found in {file_path}. Skipping.")
                continue

            filtered_table = table.filter(pc.equal(table[column_name], search_value))

            filename_array = pa.array([os.path.basename(file_path)] * filtered_table.num_rows)
            filename_field = pa.field('filename', pa.string())
            filtered_table = filtered_table.add_column(0, filename_field, filename_array)

            all_results.append(filtered_table)

        except pa.lib.ArrowTypeError as e:
            print(f"Type Error in {file_path}: {e}. Skipping file.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if all_results:
        combined_table = pa.concat_tables(all_results)
        return combined_table
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a folder of CSV files.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing CSV files")
    parser.add_argument("column_name", type=str, help="Column name to search in")
    parser.add_argument("search_value", type=str, help="Value to search for")

    args = parser.parse_args()

    result_table = search_csv_folder(args.folder_path, args.column_name, args.search_value)

    if result_table is not None:
        output_dir = "./data/mmsisearches/"  # Or wherever you want to save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "combined_search_results.csv")

        pv.write_csv(result_table, output_path)
        print(f"Filtered results saved to: {output_path}")
    else:
        print("No matching results found in the specified folder.")