import torch
from models.resnet_block import ResnetBlock
from models.conv_7 import Conv7
from einops import rearrange


class ResNet(torch.nn.Module):
    def __init__(self, num_blocks: int, embedding_dim: int):
        super().__init__()
        self.conv7 = Conv7(3, 32)

        self.resnet_blocks1 = torch.nn.Sequential(
            *[ResnetBlock(32, 32) for _ in range(num_blocks)]
        )

        self.resnet_blocks2 = torch.nn.Sequential(
            *[ResnetBlock(32, 64) for _ in range(num_blocks)]
        )

        self.resnet_blocks3 = torch.nn.Sequential(
            *[ResnetBlock(64, 128) for _ in range(num_blocks)]
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv7(x)
        out = self.resnet_blocks1(out)
        out = self.resnet_blocks2(out)
        out = self.resnet_blocks3(out)
        out = self.avg_pool(out)
        out = rearrange(
            out,
            "batch_size channels height width -> batch_size (channels height width)",
        )
        out = self.linear(out)
        return out
