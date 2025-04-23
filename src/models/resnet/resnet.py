import torch
import typing as T
from models.resnet.resnet_block import ResnetBlock
from models.resnet.conv_7 import Conv7
from einops import rearrange


class ResNet(torch.nn.Module):
    """ResNet backbone model.

    Contains a series of Resnet blocks, each with a number of modules.

    The first block is a Conv7 layer, followed by a series of Resnet blocks.

    Init args:
        num_modules_per_block: Number of modules in each Resnet block.
        block_dims: Dimensions of the Resnet blocks.
        in_channels: Number of input channels.

    """

    def __init__(
        self,
        num_modules_per_block: int,
        block_dims: T.List[int],
        in_channels: int = 1,
    ):
        super().__init__()
        self.conv7 = Conv7(in_channels, block_dims[0])

        self.resnet_blocks = torch.nn.ModuleList()

        self.resnet_blocks.append(
            torch.nn.Sequential(
                *[
                    ResnetBlock(block_dims[0], block_dims[0])
                    for _ in range(num_modules_per_block)
                ]
            )
        )

        for i in range(len(block_dims) - 1):
            in_dim = block_dims[i]
            out_dim = block_dims[i + 1]
            self.resnet_blocks.append(
                torch.nn.Sequential(
                    *[
                        ResnetBlock(in_dim, out_dim)
                        if j == 0
                        else ResnetBlock(out_dim, out_dim)
                        for j in range(num_modules_per_block)
                    ]
                )
            )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(block_dims[-1], block_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv7(x)
        for block in self.resnet_blocks:
            out = block(out)
        out = self.avg_pool(out)
        out = rearrange(
            out,
            "batch_size channels height width -> batch_size (channels height width)",
        )
        out = self.linear(out)
        return out
