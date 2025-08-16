import torch
import os
import wandb
from tqdm import tqdm
from dataset import get_loaders
import segmentation_models_pytorch as smp
from sweep_train import dice_coeff, iou_score, train_fn, eval_fn
from baseline_fcn import FCNBaseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = FCNBaseline(
    encoder_name="resnet34").to(DEVICE)


loss_fn = smp.losses.DiceLoss(mode='binary')
bce_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

wandb.init(project="polyp-baseline", config=config)



