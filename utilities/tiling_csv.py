import csv
import os

def tile_csv(input_file, output_file, y=2, x=2, out_path = '/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/tiled_csvs'):
    # Read the original CSV file into a list of rows
    with open(input_file, newline='', mode='r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    
    # Create the tiled data by repeating the rows and columns
    tiled_data = []

    # Tile rows in the y-direction (vertical repetition)
    for i in range(y):
        for row in data:
            # Tile columns in the x-direction (horizontal repetition)
            tiled_row = row * x
            tiled_data.append(tiled_row)

    # Write the tiled data to a new CSV file
    output_file = os.path.join(out_path,output_file)
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(tiled_data)

# Usage example
#tile_csv('/home/bqtx/Documents/VLSI/ir_drop_ml/utilities/mycsv.csv', 'tiled_csv.csv', y=2, x=2)