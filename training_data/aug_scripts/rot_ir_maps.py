import os
import numpy as np
import pandas as pd
from pathlib import Path

# Input and output directories
input_dir = Path("dataset/orig_maps/ir_drop_maps")
output_dir = Path("dataset/aug_maps/ir_drop_maps_rot")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the transformations
def augment_ir_map(data: np.ndarray):
    return {
        "rot90": np.rot90(data, k=1),
        "rot180": np.rot90(data, k=2),
        "rot270": np.rot90(data, k=3),
        "flip_h": np.fliplr(data),
        "flip_v": np.flipud(data),
    }

# Process each file
for file in input_dir.glob("*.csv"):
    base_name = file.stem
    data = pd.read_csv(file, header=None).values
    augmented = augment_ir_map(data)

    for suffix, arr in augmented.items():
        out_path = output_dir / f"{base_name}_{suffix}.csv"
        pd.DataFrame(arr).to_csv(out_path, header=False, index=False)
        print(f"Saved {out_path}")
