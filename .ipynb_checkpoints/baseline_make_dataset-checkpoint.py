import os
import shutil
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import h5py
import cv2
import matplotlib.pyplot as plt
import tqdm

train_metadata = pd.read_csv(os.path.join(current_dir,'train-metadata.csv'),low_memory=False)
test_metadata = pd.read_csv(os.path.join(current_dir,'test-metadata.csv'),low_memory=False)
training_validation_hdf5 = h5py.File(f"{current_dir}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{current_dir}/test-image.hdf5", 'r')