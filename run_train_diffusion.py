# %% [markdown]
# # Driver for VAE
# - We will use a Variational Autoencoder to generate more malignant data

# %%
import numpy as np
import pandas as pd
import os
import h5py
import cv2
import matplotlib.pyplot as plt
import tqdm
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
#TODO: change this to your own directory
# current_dir = "/Users/thatblue340/Documents/Documents/GitHub/EECS-545-final-project"
current_dir = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2  # now safe
import matplotlib
import matplotlib.pyplot as plt
# %%
from vae import CVAE
from torch import nn

# %% [markdown]
# ## Data Loading
# - Load images to train VAE

# %%
train_metadata = pd.read_csv(os.path.join(current_dir,'train-metadata.csv'),low_memory=False)
train_metadata.info()

# %%
test_metadata = pd.read_csv(os.path.join(current_dir,'test-metadata.csv'),low_memory=False)
test_metadata.info()

# %%
# training_validation_hdf5 = h5py.File(f"{current_dir}/train-image.hdf5", 'r')
# testing_hdf5 = h5py.File(f"{current_dir}/test-image.hdf5", 'r')
training_validation_hdf5 = h5py.File("train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"test-image.hdf5", 'r')

# %% [markdown]
# ## Preprocess data
# - Only take the malignant images
# - Resize them to (128, 128, 3)
# - Normalize pixel values to [0,1]

# %%
# import training images 
train_images = []
for i in tqdm.tqdm(range(len(train_metadata))):
    if train_metadata.iloc[i]['target'] == 0: # skip non-target images
        continue
    image_id = train_metadata.iloc[i]['isic_id']
    image = training_validation_hdf5[image_id][()]
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image / 255
    train_images.append(image)
train_images = np.array(train_images)


print(f"Training images shape: {train_images.shape}")

# %%
# plt.imshow(train_images[0])
# plt.axis('off') 
# plt.show()

# %% [markdown]
# ## Setup for Dataset to be used for diffusion model

# %%
class MalignantDataset(torch.utils.data.Dataset):
    def __init__(self, image_array):
        self.images = image_array
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                     # must come before Resize
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),                      # converts to torch.FloatTensor (float32)
            transforms.Normalize((0.5,), (0.5,))         # [-1, 1] range
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # shape: (128, 128, 3), dtype: np.uint8 or float
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image  # ensure proper dtype
        return self.transform(image)

# %%
malignant_dataset = MalignantDataset(train_images)
malignant_loader = DataLoader(malignant_dataset, batch_size=32, shuffle=True, num_workers=4)

# %% [markdown]
# ## Train using diffusion model

# %%
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

model = Unet(
    dim=128,
    dim_mults=(1, 2, 4, 8),
    channels=3
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,  # number of diffusion steps
    objective='pred_x0',  # loss objective
)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
from tqdm import tqdm
import time

model.train()
diffusion.to(device)
step = 0
max_steps = 20000
pbar = tqdm(total=max_steps, desc="Training DDPM", ncols=100)

while step < max_steps:
    for images in malignant_loader:
        images = images.to(device)
        loss = diffusion(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Step {step} | Loss: {loss.item():.4f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'diffusion_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step  # optional: useful for resume
            }, f'saved_models/diffusion_checkpoint_{step}.pth')
            # evaluate and sample
            diffusion.eval()
            with torch.no_grad():
                samples = diffusion.sample(batch_size=16)
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1).cpu()
            os.makedirs("sample_outputs", exist_ok=True)
            vutils.save_image(samples, f"sample_outputs/samples_step_{step}.png", nrow=4)
            diffusion.train()
        step += 1
        pbar.update(1)
        if step >= max_steps:
            break
pbar.close()

# %% [markdown]
# ## Generate and visualize images

# %%
# Generate 
model.eval()
diffusion.eval()

# %%
num_samples = 10
with torch.no_grad():
    samples = diffusion.sample(batch_size=num_samples)

# %%
# Convert from [-1, 1] to [0, 1]
samples = (samples + 1) / 2
samples = samples.clamp(0, 1).cpu()

# Plot images
# fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
# for i in range(num_samples):
#     img = samples[i].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
#     axes[i].imshow(img)
#     axes[i].axis("off")
# plt.tight_layout()
# plt.show()

# %%
import torchvision.transforms.functional as TF

for i, img in enumerate(samples):
    img_pil = TF.to_pil_image(img)
    img_pil.save(f"gen_image_{i:03}.png")

# %%



