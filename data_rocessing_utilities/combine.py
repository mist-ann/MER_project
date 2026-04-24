import numpy as np
import os

def concatenate_npy_files(directory, output_file):
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')],
                       key=lambda x: int(x.split('.')[0]))
    arrays = []
    for file in npy_files:
        array = np.load(os.path.join(directory, file))
        arrays.append(array)
    min_dim3 = min(array.shape[2] for array in arrays)
    resized_arrays = [array[:,:,:min_dim3] for array in arrays]
    concatenated_array = np.concatenate(resized_arrays, axis=0)

    np.save(output_file, concatenated_array)

concatenate_npy_files('music/data_scut/cochlegram', 'cochlegram.npy')
