import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from MSCF_Net import MSCF_Net


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        if file_list is None:
            self.files = sorted([f for f in os.listdir(image_dir) if f.endswith((".npy", ".npz"))])
        else:
            self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        if fname.endswith(".npz"):
            image = np.load(os.path.join(self.image_dir, fname))["arr_0"]
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

        return image, mask, fname


def compute_metrics(preds, targets):
    preds = preds.flatten()
    targets = targets.flatten()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return dice, iou, precision, recall


def test_model(data_dir, weight_path, batch_size=4, device="cuda"):
    image_dir = os.path.join(data_dir, "images", "test")
    mask_dir = os.path.join(data_dir, "masks", "test")

    dataset = SegmentationDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MSCF_Net(in_chans=1, out_chans=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    all_dice, all_iou, all_precision, all_recall = [], [], [], []

    with torch.no_grad():
        for images, masks, fnames in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            dice, iou, precision, recall = compute_metrics(preds, masks)
            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(precision)
            all_recall.append(recall)

            print(f"{fnames[0]}: Dice={dice:.4f}, IoU={iou:.4f}")

    print(f"\nFinal Results:")
    print(f"Dice:      {np.mean(all_dice):.4f} +- {np.std(all_dice):.4f}")
    print(f"IoU:       {np.mean(all_iou):.4f} +- {np.std(all_iou):.4f}")
    print(f"Precision: {np.mean(all_precision):.4f} +- {np.std(all_precision):.4f}")
    print(f"Recall:    {np.mean(all_recall):.4f} +- {np.std(all_recall):.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_model(
        data_dir="dataset",
        weight_path="mscf_net.pth",
        batch_size=1,
        device=device,
    )
