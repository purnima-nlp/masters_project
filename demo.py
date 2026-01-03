from torch.utils.data import DataLoader

from datasets.video_sr_dataset import VideoSRDataset
from datasets.pipelines import Compose, LoadFrames, RGB2Thermal, GenerateLR, ToTensor
from datasets.samplers import ClipSampler


def main():
    # Replace this with a real video path on your system
    video_list = ['sample.mp4']

    pipeline = Compose([
        LoadFrames(),
        RGB2Thermal(),
        GenerateLR(scales=[2, 3, 4], mode='random'),
        ToTensor()
    ])

    sampler = ClipSampler(clip_len=16)

    dataset = VideoSRDataset(
        video_list=video_list,
        pipeline=pipeline,
        sampler=sampler
    )

    loader = DataLoader(dataset, batch_size=1)

    lr, hr, scale = next(iter(loader))

    print(f"LR shape   : {lr.shape}")
    print(f"HR shape   : {hr.shape}")
    print(f"Scale used : x{scale.item()}")


if __name__ == "__main__":
    main()

