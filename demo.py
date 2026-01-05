import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch

from models.video_sr_model import VideoSRModel
from datasets.video_sr_dataset import VideoSRDataset
from datasets.pipelines.compose import Compose
from datasets.pipelines.loading import LoadVimeoFrames
from datasets.pipelines.transforms import RGB2Thermal, GenerateLR, ToTensor



# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("Video Super-Resolution Demo")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# -------------------------
# Main
# -------------------------
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

    # -------- Pipeline --------
    pipeline = Compose([
        LoadVimeoFrames(),
        RGB2Thermal(),
        GenerateLR(scales=[args.scale]),
        ToTensor()
    ])

    # -------- Dataset --------
    dataset = VideoSRDataset(
        root_dir=args.data_root,
        scale=args.scale,
        split="test",
        pipeline=pipeline
    )

    # Load ONE sample
    lr, hr, scale = dataset[0]
    lr_video = lr.unsqueeze(0).to(device)  # (1, C, T, H, W)

    # -------- Inference --------
    with torch.no_grad():
        sr_video = model(lr_video)

    print("LR video shape :", lr_video.shape)
    print("SR video shape :", sr_video.shape)

    # -------- Save output --------
    os.makedirs("outputs", exist_ok=True)
    torch.save(sr_video.cpu(), "outputs/sr_video.pt")
    print("[INFO] Demo finished. Output saved to outputs/sr_video.pt")


if __name__ == "__main__":
    main()


