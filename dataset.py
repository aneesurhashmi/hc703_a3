# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch 
import numpy as np
from skimage.color import rgb2gray
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_w=256, img_h=256):
    train_transform = A.Compose(
        [
        A.Resize(height=img_h, width=img_w),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )

    return train_transform, val_transforms


class CT_Dataset:
    def __init__ (self, csv_dir = "./csv/", image_set="train", transforms= None, include_background=True):
        self.transforms = transforms
        self.df = pd.read_csv(os.path.join(csv_dir, f'{image_set}.csv'))
        self.include_background = include_background

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_data = self.df.iloc[idx]
        img = plt.imread(img_data["filepath"])*255
        # img = rgb2gray(img) # 1 channel for images
        # fixing scale of masks
        liver_mask = plt.imread(img_data["liver_maskpath"])*255
        liver_mask= rgb2gray(liver_mask) # 1 channel for masks
        tumor_mask = plt.imread(img_data["tumor_maskpath"])*255
        tumor_mask= rgb2gray(tumor_mask) # 1 channel for masks
        # combined_mask  = np.where(liver_mask==1, 1, 0)
        # combined_mask  = np.where(tumor_mask==1, 2, combined_mask) # tumor class is 2

        if self.transforms is not None:
            # transformed = self.transforms(image=img, mask=combined_mask)
            transformed = self.transforms(image=img, masks=[liver_mask, tumor_mask])
            img = transformed["image"]
            # combined_mask = transformed["mask"]
            liver_mask = transformed["masks"][0]
            tumor_mask = transformed["masks"][1]
        if self.include_background:
            background_mask = torch.where(liver_mask==0, 1, 0)
            img = img[0].reshape(1, img.shape[1], img.shape[2])
            return img, torch.from_numpy(np.array([i.numpy() for i in [background_mask, liver_mask, tumor_mask]]))
        else:
            img = img[0].reshape(1, img.shape[1], img.shape[2])
            return img, torch.from_numpy(np.array([i.numpy() for i in [liver_mask, tumor_mask]]))
        # return img, combined_mask


# df = pd.read_csv("./csv/train.csv")
# df.head()

# # %%
# for i in range(len(df[df['study_number'] == 2])):
#     img_data = df[df['study_number'] == 2].iloc[i]
#     tumor_mask = plt.imread(img_data["tumor_maskpath"])
#     if  tumor_mask.sum()>2:
#         liver_mask = plt.imread(img_data["liver_maskpath"])
#         img = plt.imread(img_data["filepath"])
#         print("liver")
#         break

# # %%
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.imshow(liver_mask*255, alpha=0.3)
# plt.imshow(tumor_mask*255, alpha=0.5)



