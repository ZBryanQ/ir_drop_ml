# Extract SPICE netlist data
# TODO: either convert to CSV or apply GNN to data
# CSV strange as resistors could cross um boundaries and there are multiple layers

import re
import pandas as pd
import numpy as np
from collections import defaultdict

""" Example matches:
R645 n1_m1_108000_17920 n1_m1_102600_179200 0.14
R646 n1_m1_113400_179200 n1_M3_113400_179200 4.23
I7 n1_m1_113400_179200 0 4.24901e-08
V0 n1_m7_81000_106230 0 1.1
"""

# regex patterns
node_pattern = re.compile(r"(.*)_(.*)_(.*)_(.*)")  # sloppy, but shuld match net, layer, x, y if input is formatted correctly
resistor_pattern = re.compile(r"(R\d+) (\S+) (\S+) (\S+)")
current_source_pattern = re.compile(r"(I\d+) (\S+) (\S+) (\S+)")
voltage_source_pattern = re.compile(r"(V\d+) (\S+) (\S+) (\S+)")

class SpiceNode:
    def __init__(self, net, layer, x, y):
        self.net = net
        self.layer = layer
        self.x = x/2000 # SPICE value / 2000 is location in um
        self.y = y/2000
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

nset = set()
rset = set()
vset = set()
iset = set()

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

    
# run
spice_netlist_file = "training_data/fake-circuit-data_20230623/fake-circuit-data/current_map00.sp"
file2objects(spice_netlist_file)

# with open("./out/nodes.txt", "w") as file: 
#     [file.write(f"{node.net} {node.layer} {node.x} {node.y}\n") for node in nset]
# with open("./out/resistors.txt", "w") as file: 
#     [file.write(f"{element.name} ({element.node1.net},{element.node1.layer},{element.node1.x},{element.node1.y}) ({element.node2.net},{element.node2.layer},{element.node2.x},{element.node2.y}) {element.resistance}\n") for element in rset]
# with open("./out/voltage_sources.txt", "w") as file: 
#     [file.write(f"{element.name} ({element.node.net},{element.node.layer},{element.node.x},{element.node.y}) {element.voltage}\n") for element in vset]
# with open("./out/current_sources.txt", "w") as file: 
#     [file.write(f"{element.name} ({element.node.net},{element.node.layer},{element.node.x},{element.node.y}) {element.current}\n") for element in iset]
