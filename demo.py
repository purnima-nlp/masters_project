import sys
import os
sys.path.append(os.path.abspath('.'))
import argparse
import torch
from models.video_sr_model import VideoSRModel
from datasets.video_sr_dataset import VideoSRDataset
from pipelines.compose import Compose
from pipelines.loading import LoadVimeoFrames


# -------------------------
# Demo-only transform
# -------------------------
class DemoLRHR:
    """
    Create LR / HR tensors from loaded frames (demo only).
    """
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, results):
        frames = results["frames"]  # list of HWC numpy arrays

        # Convert to torch tensor (C, T, H, W)
        frames = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1)
            for f in frames
        ])  # (T, C, H, W)

        frames = frames.float() / 255.0
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Generate LR by downsampling
        lr = torch.nn.functional.interpolate(
            frames.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        results["lr"] = lr
        results["hr"] = frames
        return results


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
        DemoLRHR(scale=args.scale)
    ])

    # -------- Dataset --------
    dataset = VideoSRDataset(
        root_dir=args.data_root,
        scale=args.scale,
        split="test",
        pipeline=pipeline
    )

    # Get ONE sample
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

