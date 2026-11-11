import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from MSCF_Net import MSCF_Net


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2).sum(1)
        dice = (2.0 * intersection + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        return 1 - dice.mean()


class WeightedDiceCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.weight_dice * self.dice_loss(logits, targets) + self.weight_ce * self.ce_loss(logits, targets)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        if file_list is None:
            self.files = sorted([f for f in os.listdir(image_dir) if f.endswith((".npy", ".npz"))])
        else:
            self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        if fname.endswith(".npz"):
            data = np.load(os.path.join(self.image_dir, fname))
            image = data["arr_0"]
        else:
            image = np.load(os.path.join(self.image_dir, fname))
        if fname.endswith(".npz"):
            mask = np.load(os.path.join(self.mask_dir, fname))["arr_0"]
        else:
            mask = np.load(os.path.join(self.mask_dir, fname))

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, mask


def train_model(data_dir, epochs=100, batch_size=4, lr=0.0001, device="cuda"):
    train_image_dir = os.path.join(data_dir, "images", "train")
    train_mask_dir = os.path.join(data_dir, "masks", "train")
    val_image_dir = os.path.join(data_dir, "images", "val")
    val_mask_dir = os.path.join(data_dir, "masks", "val")

    train_ds = SegmentationDataset(train_image_dir, train_mask_dir)
    val_ds = SegmentationDataset(val_image_dir, val_mask_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MSCF_Net(in_chans=1, out_chans=1).to(device)
    criterion = WeightedDiceCELoss(weight_dice=0.7, weight_ce=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                dice = (2.0 * (preds * masks).sum() + 1e-6) / ((preds + masks).sum() + 1e-6)
                dice_scores.append(dice.item())

        avg_train = train_loss / len(train_ds)
        avg_val = val_loss / len(val_ds)
        avg_dice = np.mean(dice_scores)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train:.4f} Val Loss: {avg_val:.4f} Dice: {avg_dice:.4f}")

    torch.save(model.state_dict(), "mscf_net.pth")
    print("Model saved to mscf_net.pth")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_model(
        data_dir="dataset",
        epochs=100,
        batch_size=4,
        lr=0.0001,
        device=device,
    )
