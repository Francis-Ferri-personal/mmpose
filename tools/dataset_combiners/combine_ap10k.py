import json
import os
import argparse

def merge_json_splits(directory):
    """
    Merges split JSON files for train, val, and test datasets within a directory.
    Assumes file names are like:
    - ap10k-train-split1.json, ap10k-train-split2.json, ... -> ap10k-train.json
    - ap10k-val-split1.json, ap10k-val-split2.json, ... -> ap10k-val.json
    - ap10k-test-split1.json, ap10k-test-split2.json, ... -> ap10k-test.json

    Args:
        directory (str): The directory containing the split JSON files.
    """

    os.chdir(directory)  # Change to the specified directory

    datasets = ["train", "val", "test"]

    for dataset in datasets:
        file_pattern = f"ap10k-{dataset}-split"
        output_file = f"ap10k-{dataset}.json"
        merged_data = {}

        matching_files = sorted([f for f in os.listdir() if f.startswith(file_pattern) and f.endswith(".json")])

        for file_name in matching_files:
            with open(file_name, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if key in merged_data:
                        if isinstance(merged_data[key], list) and isinstance(value, list):
                            merged_data[key].extend(value)
                        else:
                            merged_data[key] = value
                    else:
                        merged_data[key] = value

        with open(output_file, 'w') as outfile:
            json.dump(merged_data, outfile)
        
        print(f"Merged {dataset} data into: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merges split JSON files for train, val, and test datasets in a directory.")
    parser.add_argument("directory", help="The directory containing the split JSON files.")
    args = parser.parse_args()

    merge_json_splits(args.directory)