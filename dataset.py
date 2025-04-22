import numpy as np
import pandas as pd
import h5py
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image

class HDF5Dataset(Dataset):
    __slots__ = ['images', 'labels', 'augment', 'transform']

    def __init__(self, images, labels, augment=False, transform=None):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert from numpy to PIL only if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)  # ensure valid range
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Directly return float tensor
        return image, torch.tensor(label, dtype=torch.float32)