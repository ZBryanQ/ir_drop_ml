# Extract SPICE netlist data
# TODO: either convert to CSV or apply GNN to data
# CSV strange as resistors could cross um boundaries and there are multiple layers

import math
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import seaborn as sns

# Globals
nset = set()
rset = set()
vset = set()
iset = set()

# regex patterns
node_pattern = re.compile(r"(.*)_(.*)_(.*)_(.*)")  # sloppy, but shuld match net, layer, x, y if input is formatted correctly
resistor_pattern = re.compile(r"(R\d+) (\S+) (\S+) (\S+)")
current_source_pattern = re.compile(r"(I\d+) (\S+) (\S+) (\S+)")
voltage_source_pattern = re.compile(r"(V\d+) (\S+) (\S+) (\S+)")

class SpiceNode:
    def __init__(self, net, layer, x, y):
        self.net = net
        self.layer = layer
        self.x = int(x)/2000 # SPICE value / 2000 is location in um
        self.y = int(y)/2000
    def __eq__(self, other) : # Same node if same net, layer, xy osition
        return (self.net, self.layer, self.x, self.y) == (other.net, other.layer, other.x, other.y)
    def __hash__(self):
        return hash((self.net, self.layer, self.x, self.y))

class SpiceResistor:
    def __init__(self, name, node1, node2, resistance):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.resistance = resistance
    def __eq__(self, other) :
        return (self.name, self.node1, self.node2, self.resistance) == (other.name, other.node1, other.node2, other.resistance)
    def __hash__(self):
        return hash((self.name, self.node1, self.node2, self.resistance))

class SpiceVsource:
    def __init__(self, name, node, voltage):
        self.name = name
        self.node = node
        self.voltage = voltage
    def __eq__(self, other):
        return (self.name, self.node, self.voltage) == (other.name, other.node, other.voltage)
    def __hash__(self):
        return hash((self.name, self.node, self.voltage))

class SpiceIsource:
    def __init__(self, name, node, current):
        self.name = name
        self.node = node
        self.current = current
    def __eq__(self, other):
        return (self.name, self.node, self.current) == (other.name, other.node, other.current)
    def __hash__(self):
        return hash((self.name, self.node, self.current))

# grid of 1 um squares
# store each um square as dict with nodes inside it 
class CSVGrid:
    def __init__(self, x_max, y_max, nset):
        pass
    
"""
Node for GNN:
-Voltage
-Current
-x
-y

Edge for GNN:
-node1
-node2
-resistance
"""

def file2objects(file_path):
    with open(file_path) as f:
        for line in f:
            r_match = re.search(resistor_pattern, line)
            v_match = re.search(voltage_source_pattern, line)
            i_match = re.search(current_source_pattern, line)

            if r_match:
                n_match_1 = re.search(node_pattern, r_match.group(2))
                n_match_2 = re.search(node_pattern, r_match.group(3))
                node1 = SpiceNode(net=n_match_1.group(1), layer=n_match_1.group(2), x=n_match_1.group(3), y=n_match_1.group(4))
                node2 = SpiceNode(net=n_match_2.group(1), layer=n_match_2.group(2), x=n_match_2.group(3), y=n_match_2.group(4))
                resistor = SpiceResistor(r_match.group(1), node1, node2, r_match.group(4))
                nset.add(node1)
                nset.add(node2)
                rset.add(resistor)

            elif v_match:
                n_match = re.search(node_pattern, v_match.group(2))
                node = SpiceNode(net=n_match.group(1), layer=n_match.group(2), x=n_match.group(3), y=n_match.group(4))
                vsource = SpiceVsource(v_match.group(1), node ,v_match.group(4))
                vset.add(vsource)

            elif i_match:
                n_match = re.search(node_pattern, i_match.group(2))
                node = SpiceNode(net=n_match.group(1), layer=n_match.group(2), x=n_match.group(3), y=n_match.group(4))
                isource = SpiceIsource(i_match.group(1), node ,i_match.group(4))
                iset.add(isource)

def get_csv_row_count(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
    return row_count

# return set of resistors that are vias
def get_vias(resistors):
    via_set = set()
    for res in rset:
        if (res.node1.x == res.node2.x) and (res.node2.y == res.node2.y) and (res.node1.layer != res.node2.layer):
            via_set.add(res)
    return via_set  # set of all resistors with same xy value but diff layers

def find_csv_size(file_path):
    csv_dir = "./training_data/csv-files/input_csvs"
    match = re.search(r'\d+', file_path) # find first number
    number_str = match.group()
    for filename in os.listdir(csv_dir):
        leading_digits_match = re.match(r'^(\d+)', filename)
        # Check if it's a CSV and contains the number
        if filename.endswith('.csv') and leading_digits_match.group(1) == number_str:
            print(f"Found matching file: {filename}")
            filepath = os.path.join(csv_dir, filename)

            # Read CSV with no headers
            df = pd.read_csv(filepath, header=None)
            num_rows = df.shape[0]
            return num_rows
    
def visualize_vias(file_path):
    # NOTE: create a different graph for each pair of layers a via bridges ex) m1 to m2, m2 to m3, etc.
    file2objects(file_path)
    via_set = get_vias(rset)
    layer_set = set()   # set of all layers found in via_set

    for via in via_set:
        layer_tuple = (via.node1.layer, via.node2.layer) # assuming direction matters instead of just using node1.layer
        if layer_tuple not in layer_set:
            layer_set.add(layer_tuple)

    for layer_tuple in layer_set: # loop through layer_set twice, not efficient? Same name for layer_tuple?
        x_coords = []
        y_coords = []
        for via in via_set:
            if layer_tuple[0] == via.node1.layer and layer_tuple[1] == via.node2.layer:
                x_coords.append(via.node1.x)
                y_coords.append(via.node1.y)

        plt.figure()
        plt.scatter(x_coords, y_coords, marker='x', s=10, color='blue')

        plt.title(f"Via from {layer_tuple[0]} to {layer_tuple[1]}")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True)

        # Save plot(s) to folder
        filename = f"{Path(file_path).stem}_{layer_tuple[0]}_{layer_tuple[1]}"
        plt.savefig(f"./temp/{filename}.png", dpi=300)  # save to file

        
    plt.show()
    
    return

def via_to_csv(file_path):
    size = find_csv_size(file_path)
    # with open('debug.txt', 'a') as file:
    #     file.write(f"CSV size: {size}\n")
    print(f"CSV size: {size}")

    file2objects(file_path)
    via_set = get_vias(rset)
    layer_set = set()   # set of all layers found in via_set

    for via in via_set: # finds n layers defined by layer_tuple
        layer_tuple = (via.node1.layer, via.node2.layer) # assuming direction matters instead of just using node1.layer
        if layer_tuple not in layer_set:
            layer_set.add(layer_tuple)

    # need n csvs, create n 2D lists of resistances per um^2

    for layer_tuple in layer_set: # loop through layer_set twice, not efficient? Same name for layer_tuple?
        resistance_matrix = [[0 for x in range(size+1)] for y in range(size+1)]
        for via in via_set:
            if layer_tuple[0] == via.node1.layer and layer_tuple[1] == via.node2.layer:
                # with open('debug.txt', 'a') as file:
                #     file.write(f"Layer {layer_tuple[0]} to {layer_tuple[1]} Node at: ({via.node1.x},{via.node1.y}), R={via.resistance} Size: {size}\n")
                resistance_matrix[math.floor(via.node1.y)][math.floor(via.node1.x)] += float(via.resistance)
        with open(f'{csv_output_dir}/{Path(file_path).stem}_{layer_tuple[0]}_{layer_tuple[1]}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(resistance_matrix)
    return

def clear_globals():
    global nset,rset,vset,iset
    nset = set()
    rset = set()
    vset = set()
    iset = set()
    
# run
csv_output_dir = "./via_csv_files"
# visualize_vias("training_data/netlists/current_map00.sp")
for i in range(100):
    clear_globals()
    s = str(i).zfill(2)
    spice_netlist_file = f"training_data/netlists/current_map{s}.sp"
    # with open('debug.txt', 'a') as file:
    #     file.write(f"Processing {spice_netlist_file}...\n")
    # visualize_vias(spice_netlist_file)
    via_to_csv(spice_netlist_file)

####################### write objects to txt ######################
# with open("./temp/nodes.txt", "w") as file: 
#     [file.write(f"{node.net} {node.layer} {node.x} {node.y}\n") for node in nset]
# with open("./temp/resistors.txt", "w") as file: 
#     [file.write(f"{element.name} ({element.node1.net},{element.node1.layer},{element.node1.x},{element.node1.y}) ({element.node2.net},{element.node2.layer},{element.node2.x},{element.node2.y}) {element.resistance}\n") for element in rset]
# with open("./temp/voltage_sources.txt", "w") as file: 
#     [file.write(f"{element.name} ({element.node.net},{element.node.layer},{element.node.x},{element.node.y}) {element.voltage}\n") for element in vset]
# with open("./temp/current_sources.txt", "w") as file: 
#     [file.write(f"{element.name} ({element.node.net},{element.node.layer},{element.node.x},{element.node.y}) {element.current}\n") for element in iset]
