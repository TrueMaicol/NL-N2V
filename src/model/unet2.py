import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class MaxBlurPool2D(nn.Module):
  """
  Class that constructs the MaxBlurPool layer. 
  For now only kernel size 3x3 and 5x5 has been implemented. 

    Args:
      - pool_size (int): (default=2)
  """
  def __init__(self, pool_size:int=2, kernel_size:int=3):
    super(MaxBlurPool2D, self).__init__()
    self.pool_size = pool_size
    self.kernel_size = kernel_size

    # Generate the Gaussian blurring kernel
    if kernel_size == 3:
      blur_kernel = torch.tensor([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]], dtype=torch.float32)
      blur_kernel = blur_kernel / torch.sum(blur_kernel)
    elif kernel_size == 5:
      blur_kernel = torch.tensor([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 26, 4],
                                  [6, 24, 36, 24, 6],
                                  [4, 16, 24, 26, 4],
                                  [1, 4, 6, 4, 1]], dtype=torch.float32)
      blur_kernel = blur_kernel / torch.sum(blur_kernel)
    else: 
      raise NotImplementedError(f"Kernel size not yet implemented, only 3/5 are impremented for now.")
    
    # Register the kernel as a buffer
    self.register_buffer('blur_kernel', blur_kernel.view(1, 1, kernel_size, kernel_size))
  
  def forward(self, x):
    """
    Forward method for the MaxBlurPool2D class. 
    It performs maxpooling then blurs the result by convolving the Gaussian kernel.
    """
    # Apply max pooling with stride 1 and same padding
    x = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

    # Depthwise convolution with the blur kernel
    batch_size, channels, height, width = x.shape
    x = x.view(1, batch_size * channels, height, width)
    x = F.conv2d(x, self.blur_kernel.expand(batch_size * channels, 1, self.kernel_size, self.kernel_size), stride=1, padding=self.kernel_size // 2, groups=batch_size * channels)
    x = x.view(batch_size, channels, x.shape[2], x.shape[3])

    return x

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

class UNet2(nn.Module):
  """
  Unet based on the one given by the article: "N2V2 - Fixing Noise2Void chekerboard artifcats with modified sampling strategies and a tweaked network architecture".
  It is a 3 level Unet, without the first level's skip connection and with downsampling performed through MaxBlurPooling operation. 

    Args: 
      - in_channels: number of input channels (default=3)
      - out_channels: number of output channels (default=3)
      - features: least of feature produced by each double convolution layer. (default=[96, 192])
  """

  def __init__(self, in_channels=3, out_channels=3, features=[96, 192]):
    super().__init__()

    self.down_branch = nn.ModuleList()
    self.up_branch = nn.ModuleList()

    self.pool = MaxBlurPool2D()

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
    self.up_branch.append(UpConv(features[-1]*2, features[-1]))
    self.up_branch.append(DoubleConv(features[-1]*2, features[-1]))
    self.up_branch.append(UpConv(features[-2]*2, features[-2]))
    self.up_branch.append(DoubleConv(features[-2], features[-2]))
    
    self.bottom_layer = DoubleConv(features[-1], features[-1]*2)

    # The last layers, following B2U, are 3 1x1 convolutional layers
    self.tail = nn.Sequential(
      DoubleConv(features[0], features[0], kernel_size=1, padding=0),
      nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    )

  def forward(self, x):
    dim = x.shape
    # Add the down-branch
    x1 = self.down_branch[0](x)
    x2 = self.pool(x1)
    x2 = self.down_branch[1](x2)
    skip_connection = x2
    x3 = self.pool(x2)

    # Add the bottom layer
    x_bottom = self.bottom_layer(x3)
    
    # Add the up-branch
    x = self.up_branch[0](x_bottom)
    # Concat the skip connection, checking the dimension
    if x.shape != skip_connection.shape:
      x = TF.resize(x, size=skip_connection.shape[2:])
    
    concat_skip = torch.cat((skip_connection, x), dim=1)
    # Perform convolution
    x = self.up_branch[1](concat_skip)
    x = self.up_branch[2](x)
    x = self.up_branch[3](x)

    # Add tail 
    out = self.tail(x)
    if out.shape != dim:
      out = TF.resize(out, size=dim[2:])

    return out

def test():
  rand_shape = (4, 1, 321, 481)
  x = torch.randn(rand_shape)

  model = UNet2(in_channels=rand_shape[1], out_channels=rand_shape[1])
  pred = model(x)

  print(f"input shape: {x.shape}")
  print(f"output shape: {pred.shape}")
  assert x.shape == pred.shape, "The network doesn't produce the same dimensions"

if __name__ == '__main__':
  test()