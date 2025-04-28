import os
import shutil

# Base paths
orig_spice_dir = "dataset/orig_maps/spice_files"
aug_csv_dir = "dataset/aug_maps/current_maps_scale"
output_spice_dir = "dataset/aug_maps/spice_files"

# Ensure output directory exists
os.makedirs(output_spice_dir, exist_ok=True)

# Define augmentation suffixes to look for
augmentations = ["scaled_0.9", "scaled_1.1", "noisy"]

# Loop through all original spice files
for i in range(100):
    base_name = f"current_map{i:02d}"
    orig_sp_path = os.path.join(orig_spice_dir, f"{base_name}.sp")

    if not os.path.exists(orig_sp_path):
        print(f"Original SPICE file missing: {orig_sp_path}")
        continue

    for aug in augmentations:
        aug_csv_filename = f"{base_name}_current_{aug}.csv"
        aug_csv_path = os.path.join(aug_csv_dir, aug_csv_filename)

        if not os.path.exists(aug_csv_path):
            print(f"Augmented CSV missing: {aug_csv_path}")
            continue

        new_sp_name = f"{base_name}_{aug}.sp"
        new_sp_path = os.path.join(output_spice_dir, new_sp_name)

        shutil.copyfile(orig_sp_path, new_sp_path)
        print(f"Created: {new_sp_path}")
