"""
test-train split as per the ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv file
"""
import argparse
import csv
import json
import os


def process_csv(file_path, output_dir):
    # Dictionary to hold image paths categorized by split partition
    split_partitions = {}

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            included = row['included_in_public_release'].strip().lower() == 'true'
            exclude_reason = row['exclude_reason'].strip()
            split_partition = row['split_partition_name'].strip()

            # Check if image meets the criteria
            if included and not exclude_reason and split_partition:
                image_path = os.path.join("downloads",
                    row['scene_name'], "images", f"scene_{row['camera_name']}_final_preview",
                    f"frame.{row['frame_id'].zfill(4)}.tonemap.jpg"
                )

                depth_path = os.path.join("downloads",
                    row['scene_name'], "images", f"scene_{row['camera_name']}_geometry_hdf5",
                    f"frame.{row['frame_id'].zfill(4)}.depth_meters.hdf5"
                )

                # Store the image path in the corresponding split partition
                if split_partition not in split_partitions:
                    split_partitions[split_partition] = []
                split_partitions[split_partition].append((image_path, depth_path))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the image paths to separate text files based on split partition
    for split, paths in split_partitions.items():
        output_file = os.path.join(output_dir, f"{split}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for pt in paths:
                f.write(json.dumps(list(pt)) + '\n')  # Convert tuple to list for json serialization

    print("Processing complete. Split files saved in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_path", type=str, help="path of the csv defining the dataset split")
    parser.add_argument("-op_dir", type=str, help="directory to save the output txt files")
    args = parser.parse_args()

    process_csv(args.csv_path, args.op_dir)
