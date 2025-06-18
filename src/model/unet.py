import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
  """
  This class introduces a double convolution layer. It is used in each level of the UNet, both in the down branch and in the up one.

    Args:
      - in_channels: Number of input channels to the first of the 2 convolutions
      - out_channels: Number of the output channels of the double convolution
      - kernel_size: size of the kernel applied (default=3)
      - stride: stride used when applying the kernel (default=1)
      - padding: padding used in the conv2d operation (default=1)
  """

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      # nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      # nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True), 
    )
  
  def forward(self, x):
    return self.conv(x)
  

class UpConv(nn.Module):
  """
  This class is used as an alternative to the nn.ConvTranspose2d module of torch, which can introduce checkerboard artifacts in the resulting images.

    Args: 
      - in_channels: Number of channels of input to the upsampling layer
      - out_channels: Number of the output channels coming out of the upsampled layer
  """
  def __init__(self, in_channels, out_channels):
    super(UpConv, self).__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, x):
    x = self.upsample(x)
    x = self.conv(x)
    x = self.relu(x)
    return x


class UNet(nn.Module):
  """
  This class builds the UNet architecture following the original UNet paper.

    Args:
      - in_channels: Number of channels of the images in input to the network (default = 3)
      - out_channels: Number of channels of the images the network outputs (default = 3)
      - features: List of feature produced by each double convolution layer (default: [64, 128, 256, 512])
  """
  def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
    super().__init__()

    self.down_branch = nn.ModuleList()
    self.up_branch = nn.ModuleList()

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # transform the str elements of the array features into integer values
    features = list(map(int, features))
    for feat in features:
      assert isinstance(feat, int), f"Features array doesn't contain integer values!"

    # make sure in_channels and out_channels are integer values
    if isinstance(in_channels, str): 
      in_channels = int(in_channels)
    if isinstance(out_channels, str):
      out_channels = int(out_channels)

    # Setup the DOWN branch
    for feature in features:
      # Append a double conv layer with in_channels set at first as the input of the UNet
      self.down_branch.append(DoubleConv(in_channels, feature))
      # Change the input channels accordingly to the one produced in output by the double convolution
      in_channels = feature
    
    # Setup the UP branch
    for feature in reversed(features):
      self.up_branch.append(
        UpConv(feature*2, feature)
      )
      self.up_branch.append(DoubleConv(feature*2, feature))
    
    self.bottom_layer = DoubleConv(features[-1], features[-1]*2)

    # The last layers, following B2U, are 3 1x1 convolutional layers
    self.tail = nn.Sequential(
      DoubleConv(features[0], features[0], kernel_size=1, padding=0),
      nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    )

  def forward(self, x):
    skip_connections = []

    # Add the down_branch
    for down in self.down_branch:
      # perform double convolution and save layer for the skip connection
      x = down(x)
      skip_connections.append(x)
      # Perform maxpooling
      x = self.pool(x)
    
    # Add the bottom layer
    x = self.bottom_layer(x)
    
    # Add the up_branch
    skip_connections = skip_connections[::-1]
    for idx in range(0, len(self.up_branch), 2):
      # Perform the upsampling
      x = self.up_branch[idx](x)
      # Add the skip connection
      skip_connection = skip_connections[idx//2]
      
      # Resize the shape of the maxpooled image to handle sizes that can't be divided by 2
      if x.shape != skip_connection.shape: 
        x = TF.resize(x, size=skip_connection.shape[2:]) 
      
      concat_skip = torch.cat((skip_connection, x), dim=1)
      # Perform the double convolution
      x = self.up_branch[idx+1](concat_skip)
    
    # Add the final 3 (1x1) convolutions
    out = self.tail(x)

    return out


# Simple testing function for the shapes
def test_shapes():
  rand_shape = (4, 3, 250, 250)
  x = torch.randn(rand_shape)

  model = UNet(in_channels=rand_shape[1], out_channels=rand_shape[1])
  preds = model(x)

  print(f"preds shape: {preds.shape}")
  print(f"input shape: {x.shape}")
  assert x.shape == preds.shape, "Something wrong with the dimensions!"


if __name__ == "__main__":
  test_shapes()