import os
import numpy as np
import pandas as pd

# Augmentation functions
def add_noise(data, noise_level=0.03):
    noise = np.random.normal(0, noise_level * np.abs(data), data.shape)
    return data + noise

def scale_data(data, factor):
    return data * factor

# Directories
input_base = './dataset/orig_maps'
output_base = './dataset/aug_maps'

# Only augment these maps
map_types = ['current_maps', 'eff_dist_maps']
noise_level = 0.03
scale_factors = [0.9, 1.1]

# Make output directories
for map_type in map_types:
    out_dir = os.path.join(output_base, map_type)
    os.makedirs(out_dir, exist_ok=True)

# Process each map type
for map_type in map_types:
    in_dir = os.path.join(input_base, map_type)
    out_dir = os.path.join(output_base, map_type)

    print(f'Augmenting: {map_type}')

    for file_name in os.listdir(in_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(in_dir, file_name)
            data = pd.read_csv(file_path, header=None).values

            # Save original (optional)
            orig_out = os.path.join(out_dir, f'{file_name[:-4]}_orig.csv')
            pd.DataFrame(data).to_csv(orig_out, header=False, index=False)

            # Noisy version
            noisy = add_noise(data, noise_level)
            noisy_out = os.path.join(out_dir, f'{file_name[:-4]}_noisy.csv')
            pd.DataFrame(noisy).to_csv(noisy_out, header=False, index=False)

            # Scaled versions
            for scale in scale_factors:
                scaled = scale_data(data, scale)
                scaled_out = os.path.join(out_dir, f'{file_name[:-4]}_scaled_{scale}.csv')
                pd.DataFrame(scaled).to_csv(scaled_out, header=False, index=False)

print("complete")
