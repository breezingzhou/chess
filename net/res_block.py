import torch.nn as nn


class ResBlock(nn.Module):

  def __init__(self, num_filters=256):
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_filters),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_filters),
    )

  def forward(self, x):
    y = self.conv1(x)
    y = self.conv2(y)
    y += x
    return nn.ReLU()(y)
