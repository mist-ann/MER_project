import numpy as np
import csv


label_a = []
label_v = []

with open('music/data_scut/MEL_preprocess/arousal_cont_average.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        label_a.extend([float(i) for i in row[1:]])

with open('music/data_scut/MEL_preprocess/valence_cont_average.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        label_v.extend([float(i) for i in row[1:]])

data_a = np.array(label_a)
np.save('label_a.npy', data_a)
data_v = np.array(label_v)
np.save('label_v.npy', data_v)