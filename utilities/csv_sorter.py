import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import re
import os
import shutil
from tiling_csv import tile_csv

def get_og_csv(fake_data_dir="/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/fake-circuit-data_20230623", real_data_dir="/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/real-circuit-data_20230615"):
    #path to fake data directory; eg fake_data_dir = "./fake-circuit-data_20230623"
    # path to real data directory; real_data_dir = "./real-circuit-data_20230615"
    #dic1 stores all of the files I will be modifying together
    list = []
    for path, subdirs, files in os.walk(fake_data_dir):
        for name in files:
            if "csv" in name:
                list.append(os.path.join(path,name))
    for subdirs in os.listdir(real_data_dir):
        subdir_path = os.path.join(real_data_dir,subdirs) 
        for files in os.listdir(subdir_path):
            if "csv" in files:
                list.append(os.path.join(subdir_path, files))
    return list

def set_storage_path_png(file_path, output_dir):
    if "testcase" not in file_path:
        output_path = re.split(r'\/',file_path)[-1]
        output_path = re.split(r".csv$", output_path)[0]
        output_path = re.sub("current_map", "", output_path)
        output_path = output_path + ".csv"
    else:
        testcase_num = int(re.split("testcase",re.findall(r"testcase\d+", file_path).pop())[-1])
        output_path = re.split(r'\/',file_path)[-1]
        output_path = re.split(r".csv$", output_path)[0]
        output_path = str(testcase_num+100) + "_" + output_path + ".csv"
    output_path = os.path.join(output_dir, output_path)    
    return output_path


#this function doesnt totally work with the real circuit testcases btw
def sort_png_files(dir_path = '/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/csv-files'):
    list = os.listdir(dir_path)
    for f in list:
        file = os.path.join(dir_path, f)
        if os.path.isdir(file):
            continue
            print('continue')
        else:
            if "ir_drop" in f:
                shutil.move(file, os.path.join(dir_path,"ir_drop_csvs",f))
            else:
                shutil.move(file, os.path.join(dir_path,"input_csvs",f))

def csv2png(file_path, output_dir='/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/csv-files/'):
    #read_csv returns a two-dimensional DataFrame or TextFileReader with labeled axes
    arr_2D = pd.read_csv(file_path)
    image = arr_2D.to_numpy()
    np.savetxt(set_storage_path_png(file_path,output_dir), image, delimiter=',')

def csvs2pngs():
    csvs = get_og_csv()
    for csv in csvs:
        with open(csv) as f:
            first_line = f.readline()
            if len(first_line.split(",")) > 930/2:
                csv2png(csv)
            else:
                print(csv)
                output_file = os.path.split(csv)[-1]
                tile_csv(csv, output_file, 2, 2, out_path = '/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/tiled_csvs')
    for file in os.listdir('/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/tiled_csvs'):
        csv2png(os.path.join('/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/tiled_csvs', file))

def main():
    # csvs2pngs()
    sort_png_files()

if __name__ == '__main__':
    main()