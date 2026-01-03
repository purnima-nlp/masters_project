import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- your modules ----
from datasets.video_sr_dataset import VideoSRDataset
from models.video_sr_model import VideoSRModel
from losses.L1 import L1Loss


def parse_args():
    parser = argparse.ArgumentParser("Video Super-Resolution Training")

    parser.add_argument("--train_root", type=str, required=True,
                        help="Path to training dataset root")
    parser.add_argument("--scale", type=int, default=4,
                        help="Super-resolution scale factor")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    # -------- Dataset --------
    train_dataset = VideoSRDataset(
        root_dir=args.train_root,
        scale=args.scale,
        split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # -------- Model --------
    model = VideoSRModel(scale=args.scale)
    model = model.to(device)

    # -------- Loss --------
    criterion = L1Loss()

    # -------- Optimizer --------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # -------- Training Loop --------
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]")

        for batch in pbar:
            lr_video = batch["lr"].to(device)   # (B, C, T, H, W)
            hr_video = batch["hr"].to(device)

            optimizer.zero_grad()

            sr_video = model(lr_video)
            loss = criterion(sr_video, hr_video)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

        # -------- Save Checkpoint --------
        ckpt_path = os.path.join(
            args.save_dir, f"epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path
        )

    print("Training finished.")


if __name__ == "__main__":
    main()

