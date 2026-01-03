import argparse
import torch
import os

from models.video_sr_model import VideoSRModel
from datasets.video_sr_dataset import VideoSRDataset


def parse_args():
    parser = argparse.ArgumentParser("Video Super-Resolution Demo")

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root (or folder containing videos)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pth). If not provided, runs with random weights."
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Super-resolution scale factor"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # -------- Model --------
    model = VideoSRModel(scale=args.scale)

    if args.checkpoint is not None:
        print(f"[INFO] Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
    else:
        print("[INFO] No checkpoint provided. Using randomly initialized weights.")

    model = model.to(device)
    model.eval()

    # -------- Load one sample --------
    dataset = VideoSRDataset(
        root_dir=args.data_root,
        scale=args.scale,
        split="test"
    )

    sample = dataset[0]
    lr_video = sample["lr"].unsqueeze(0).to(device)  # (1, C, T, H, W)

    # -------- Inference --------
    with torch.no_grad():
        sr_video = model(lr_video)

    print("LR video shape :", lr_video.shape)
    print("SR video shape :", sr_video.shape)

    # -------- Save output tensor --------
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/sr_video.pt"
    torch.save(sr_video.cpu(), output_path)

    print(f"[INFO] Demo finished. Output saved to {output_path}")


if __name__ == "__main__":
    main()

