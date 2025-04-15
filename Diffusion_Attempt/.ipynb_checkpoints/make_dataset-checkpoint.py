import os
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


os.makedirs("data/train/benign", exist_ok=True)
os.makedirs("data/train/malignant", exist_ok=True)
current_dir = ""
train_hdf5 = h5py.File("train-image.hdf5", "r")
train_metadata = pd.read_csv(os.path.join(current_dir,'train-metadata.csv'),low_memory=False)   

for i in tqdm(range(len(train_metadata))):
    row = train_metadata.iloc[i]
    image_id = row["isic_id"]
    label = row["target"]
    
    image = train_hdf5[image_id][()]
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))

    out_path = f"data/train/{'malignant' if label == 1 else 'benign'}/{image_id}.png"
    cv2.imwrite(out_path, image)