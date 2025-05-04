import os
import h5py
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm import tqdm
import cv2

n_augmentations = 5
augmented_hdf5_path = 'augmented_data.hdf5'
augmented_metadata_path = 'augmented_metadata.csv'

original_hdf5_path = 'train-image.hdf5'
original_metadata_path = 'train-metadata.csv'

train_metadata = pd.read_csv(original_metadata_path,low_memory=False)   
train_hdf5 = h5py.File(original_hdf5_path, 'r')

train_images = []
for i in tqdm(range(len(train_metadata))):
    if train_metadata.iloc[i]['target'] == 0: # skip non-target images
        continue
    image_id = train_metadata.iloc[i]['isic_id']
    image = train_hdf5[image_id][()]
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image / 255
    train_images.append(image)
train_images = np.array(train_images)


print(f"Training images shape: {train_images.shape}")

assert train_images.shape[1:] == (128, 128, 3), "Incorrect image shape detected."

augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()  # back to tensor (0–1 float)
])

hdf5_file = h5py.File(augmented_hdf5_path, "w")
augmented_metadata = []

for idx, img in tqdm(enumerate(train_images), total=len(train_images), desc="Augmenting"):
    orig_isic_id = train_metadata.iloc[idx]['isic_id']
    target = train_metadata.iloc[idx]['target']
    img_uint8 = (img * 255).astype("uint8")
    
    for j in range(n_augmentations):
        aug_tensor = augmentation(img_uint8)  # Tensor CxHxW
        aug_img_np = (aug_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        new_id = f"{orig_isic_id}_aug{j}"
        hdf5_file.create_dataset(new_id, data=aug_img_np, dtype='uint8')

        augmented_metadata.append(train_metadata.iloc[idx].to_dict())
        
hdf5_file.close()
augmented_metadata_df = pd.DataFrame(augmented_metadata)
augmented_metadata_df.to_csv(augmented_metadata_path, index=False)
print(f"Augmented metadata saved to {augmented_metadata_path}")

    
    
