import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from dataset import get_loaders
import segmentation_models_pytorch as smp

# CONFIG 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SAVE_MODEL = True

def dice_coeff(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = (preds + targets - preds * targets).sum()
    return intersection / (union + 1e-8)

def train_fn(loader, model, optimizer, loss_fn, bce_loss):
    model.train()
    loop = tqdm(loader)
    total_loss, total_dice, total_iou = 0, 0, 0

    for imgs, masks in loop:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = bce_loss(preds, masks) + loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = dice_coeff(preds, masks).item()
        iou = iou_score(preds, masks).item()

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        loop.set_postfix(loss=loss.item(), dice=dice, iou=iou)

    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)

def eval_fn(loader, model):
    model.eval()
    total_dice, total_iou = 0, 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)

            total_dice += dice_coeff(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

    return total_dice / len(loader), total_iou / len(loader)

# SWEEP TRAIN FUNCTION 
def sweep_train():
    wandb.init(project="polyp-trial2")
    config = wandb.config

    train_loader, val_loader = get_loaders(
        batch_size=config.batch_size,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH
    )

    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_channels=config.decoder_channels,
        encoder_depth=config.encoder_depth,
    ).to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_dice = 0.0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_dice, train_iou = train_fn(train_loader, model, optimizer, loss_fn, bce_loss)
        val_dice, val_iou = eval_fn(val_loader, model)

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Dice": train_dice,
            "Train IoU": train_iou,
            "Val Dice": val_dice,
            "Val IoU": val_iou
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New Best Model Saved | Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            wandb.summary["Best Dice"] = val_dice
            wandb.summary["Best IoU"] = val_iou

#Manual testing (optional)
if __name__ == "__main__":
    wandb.init(project="polyp-trial2", config={
        "epochs": 5,
        "lr": 1e-4,
        "batch_size": 8,
        "encoder_name": "resnet34",
        "decoder_channels": [256, 128, 64, 32, 16],
        "encoder_depth": 5
    })
    sweep_train()
