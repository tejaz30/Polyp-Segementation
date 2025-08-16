from glob import glob
import os
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2
import torch

class PolypDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)  # grayscale
        mask = (mask > 0).astype('float32')    # binary

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0) # shape: (1, H, W)


        return img, mask

def get_loaders(batch_size=8, img_height=256, img_width=256, data_dir="data/Kvasir-SEG"):
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    images = sorted(glob(os.path.join(image_dir, "*.jpg")))
    masks = sorted(glob(os.path.join(mask_dir, "*.jpg")))

    images = sorted(glob(os.path.join(image_dir, "*.jpg")))
    masks = sorted(glob(os.path.join(mask_dir, "*.jpg")))

    print("Images found:", len(images))
    print("Masks found:", len(masks))
    print("Image path example:", images[0] if images else "None")
    print("Mask path example:", masks[0] if masks else "None")

    if len(images) == 0 or len(masks) == 0:
        raise ValueError(f"No images or masks found in {image_dir} and {mask_dir}. Check file paths and extensions.")


    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )

    train_transform = A.Compose([
        A.Resize(img_height, img_width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    train_dataset = PolypDataset(train_images, train_masks, transform=train_transform)
    val_dataset = PolypDataset(val_images, val_masks, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
