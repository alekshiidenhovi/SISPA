import torch


class ResnetBlock(torch.nn.Module):
    """A residual block for ResNet architectures.

    This block performs two convolutions with batch normalization and ReLU activation,
    followed by a residual connection that adds the input to the result.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, optional
        Size of the convolving kernel, by default 3
    stride : int, optional
        Stride of the convolution, by default 1
    padding : int, optional
        Zero-padding added to both sides of the input, by default 1

    Attributes
    ----------
    conv1 : torch.nn.Conv2d
        First convolutional layer
    batch_norm1 : torch.nn.BatchNorm2d
        First batch normalization layer
    relu : torch.nn.ReLU
        ReLU activation function
    conv2 : torch.nn.Conv2d
        Second convolutional layer
    batch_norm2 : torch.nn.BatchNorm2d
        Second batch normalization layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding
        )
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        out1 = self.batch_norm1(self.conv1(x))
        out_relu1 = self.relu(out1)
        out2 = self.batch_norm2(self.conv2(out_relu1))
        return self.relu(out2 + x)
