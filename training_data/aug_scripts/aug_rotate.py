import os
import numpy as np
import pandas as pd
from pathlib import Path

# Root directory containing original maps
orig_root = Path("dataset/orig_maps")
# Output directory for augmented maps
aug_root = Path("dataset/aug_maps")

# Define the map categories and their subdirectories
map_categories = {
    "current": "current_maps",
    "eff_dist": "eff_dist_maps",
    "pdn_density": "pdn_density_maps"
}

# Define augmentation transformations
augmentations = {
    "rot90": lambda x: np.rot90(x, k=1),
    "rot180": lambda x: np.rot90(x, k=2),
    "rot270": lambda x: np.rot90(x, k=3),
    "flip_horizontal": lambda x: np.fliplr(x),
    "flip_vertical": lambda x: np.flipud(x)
}

def augment_maps():
    for name, subdir in map_categories.items():
        input_dir = orig_root / subdir
        output_dir = aug_root / f"{subdir}_rot"
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in input_dir.glob("*.csv"):
            basename = file.stem
            data = pd.read_csv(file, header=None).values

            for aug_name, transform in augmentations.items():
                aug_data = transform(data)
                aug_filename = f"{basename}_{aug_name}.csv"
                pd.DataFrame(aug_data).to_csv(output_dir / aug_filename, index=False, header=False)

            print(f"Augmented: {file.name} -> {output_dir}")

if __name__ == "__main__":
    augment_maps()
