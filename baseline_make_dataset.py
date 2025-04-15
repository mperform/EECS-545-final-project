import os
import shutil
import random
from sklearn.model_selection import train_test_split

train_metadata = pd.read_csv(os.path.join(current_dir,'train-metadata.csv'),low_memory=False)
test_metadata = pd.read_csv(os.path.join(current_dir,'test-metadata.csv'),low_memory=False)
training_validation_hdf5 = h5py.File(f"{current_dir}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{current_dir}/test-image.hdf5", 'r')