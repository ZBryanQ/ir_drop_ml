# Put CSV into excel sheet with each cell colored from green to red normalized to the min and max of the CSV
import pandas as pd
from pathlib import Path

input_file = 'fake-circuit-data/current_map00_current.csv'  # Change this to desired input
output_file = f'{Path(input_file).stem}_colored.xlsx'

df = pd.read_csv(input_file, header=None)

min_val = df.min().min()
max_val = df.max().max()

def color_map(val):
    green = int((1 - val) * 255)
    red = int(val * 255)
    color = f'#{red:02X}{green:02X}00'  # Format as hex: #RRGGBB
    return f'background-color: {color}'

# Apply color based on normalized value
styled_df = df.style.map(lambda val: color_map((val - min_val) / (max_val - min_val)))
# Output
styled_df.to_excel(output_file, engine='openpyxl', index=False, header=False)
