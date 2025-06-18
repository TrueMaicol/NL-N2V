import torch
import torch.nn as nn

from ..model.unet import UNet
from ..model.unet2 import UNet2

class NL_BSN(nn.Module):
  """
  This class is used to initialize all the parameter of the model. Moreover, it is used to perform 
  pixel shuffle down sampling, masking and predictions
  """
  def __init__(self, pd_training:int=5, pd_testing:int=2, bsn:str='UNet', in_channels:int=3, features=[64, 128, 256, 512]):
    """
    Initializer for the denoiser
      Args:
          pd_training (int): stride factor for the pixel_shuffle operation during training
          pd_testing (int): stride factor for the pixel_shuffle operation during testing
          bsn (str): name of the used network 
          in_channels (int): number of input channels to the network
          features : list containing the number of features used by the network on each level
    """
    super().__init__()

    # DEBUGGING

    # print(f'bsn: {bsn}, in_channels: {in_channels}')
    # print('features:')
    # for feat in features: 
    #   print(type(feat), feat)

    # set the network hyperparameters
    self.pd_tr = int(pd_training)
    self.pd_ts = int(pd_testing)
    
    if bsn == 'UNet':
      self.bsn = UNet(int(in_channels), int(in_channels), features)
    elif bsn == 'UNet2':
      # Since the implementation is not parametric, check the number of features
      assert len(features) == 2, f"Num of features is too big for the UNet2 implementation! Expected 2, got: {len(features)}"
      self.bsn = UNet2(int(in_channels), int(in_channels), features)
    else: 
      raise NotImplementedError("Unknown bsn name")
    
  def get_pd_training(self) -> int:
    """
    Getter for the stride factor to use during training
    """
    return self.pd_tr

  def get_pd_testing(self) -> int:
    """
    Getter for the stride factor to use during testing
    """
    return self.pd_ts

  def forward(self, img:torch.Tensor):
    """
    Necessary forward function for the model. It calls the forward of the UNet.

      Args:
        - img (torch.Tensor): batch of images to denoise.
      
      Returns:
        - denoised batch coming out of the UNet.
    """
    # print(f'input image shape: {img.shape}')
    return self.bsn(img)
    
