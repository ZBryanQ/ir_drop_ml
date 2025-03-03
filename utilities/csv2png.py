import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np

def get_training_data(fake_data_dir="./../training_data/fake-circuit-data_20230623", real_data_dir="./../training_data/real-circuit-data_20230615")
    #path to fake data directory; eg fake_data_dir = "./fake-circuit-data_20230623"
    # path to real data directory; real_data_dir = "./real-circuit-data_20230615"
    #dic1 stores all of the files I will be modifying together
    dic1 = {}
    for path, subdirs, files in os.walk(fake_data_dir):
        for name in files:
            if "csv" in name:
                key = re.findall(r"\d\d", name).pop()
                if key not in dic1:
                    dic1[key] = []
                dic1[key].append(os.path.join(path,name))
    for subdirs in os.listdir(real_data_dir):
        subdir_path = os.path.join(real_data_dir,subdirs) 
        if subdirs not in dic1:
            dic1[subdirs] = []
        for files in os.listdir(subdir_path):
            if "csv" in files:
                dic1[subdirs].append(os.path.join(subdir_path, files))

def csv2png(file_path)
    