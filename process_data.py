import os
import argparse
import pandas as pd
import pickle as pkl
import numpy as np

# for each file in the folder raw_data, we want to load in and rescale the data to be in kilowats per minute
# we can optionally reformat the data to be a table of size 1440 x 365 (minute x day).


parser = argparse.ArgumentParser()
parser.add_argument("-r","--raw-path", default=os.path.join("data","raw_data"))
parser.add_argument("-d", "--dump-path", default= os.path.join("data","processed_data"))
args = parser.parse_args()

for file in os.listdir(args.raw_path):

    file_path = os.path.join(args.raw_path,file)
    data = pd.read_csv(file_path, header=None).to_numpy()

    minutes, _ = data.shape
    days = (minutes // 60) // 24
    minutes_per_day = minutes // days
    hours = minutes_per_day // 60 

    data = np.reshape(data, (minutes_per_day, days))
    binned_data = np.zeros(( hours, days))

    for i in range(hours):
        first, last = i * 60, (i+1) * 60
        print(data[first:last])
        binned_data[i] = np.mean( data[first:last], axis=0) / 10 **3 
    
    print(binned_data)

    file_name = file.split(".")[0]
    pickle_path = os.path.join(args.dump_path, file_name + ".pkl")
    pkl.dump(
        binned_data,
        open(pickle_path,"wb")
    )