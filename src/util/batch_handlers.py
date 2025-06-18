import torch
import torch.nn.functional as F

# These 2 first functions are taken from "AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network"
# The functions in this file are all used in case of a downsampling operation is introduced in the pipeline.

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
  """
  Pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    
    Args:
      - x (Tensor) : input tensor
      - f (int) : factor of PD
      - pad (int) : number of pad between each down-sampled images
      - pad_value (float) : padding value
    
    Return:
      - pd_x (Tensor) : down-shuffled image tensor with pad or not
  """
  # single image tensor
  if len(x.shape) == 3:
    c,w,h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    return unshuffled.view(c, f, f, w//f+2*pad, h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
  # batched image tensor
  else:
    b,c,w,h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    return unshuffled.view(b, c, f, f, w//f+2*pad, h//f+2*pad).permute(0,1,2,4,3,5).reshape(b, c, w+2*f*pad, h+2*f*pad)


def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
  """
  Inverse of pixel-shuffle down-sampling (PD) see more details about PD in pixel_shuffle_down_sampling()
    
    Args:
      - x (Tensor) : input tensor
      - f (int) : factor of PD
      - pad (int) : number of pad will be removed
  """
  #single image tensor
  if len(x.shape) == 3:
    c,w,h = x.shape
    before_shuffle = x.view(c, f, w//f, f, h//f).permute(0, 1, 3, 2, 4).reshape(c*f*f, w//f, h//f)
    if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)
  # batched image tensor
  else:
    b,c,w,h = x.shape
    before_shuffle = x.view(b, c, f, w//f, f, h//f).permute(0, 1, 2, 4, 3, 5).reshape(b, c*f*f, w//f, h//f)
    if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)


def reshape_batch(batch:torch.Tensor, sf:int):
  """
  Function that gets in input a batch of images with the following shape (b, c, h, w), where b and c are
  the batch size and the number of channels, and a stride factor.
  Then splits the images into sub-patches of size (h//sf, w//sf) and rearrange them together in a new batch.
    
    Args:
      - batch (torch.Tensor): bacth of images of shape (b, c, h, w)
      - sf (int): stride factor applied to those images

    Returns:
      - smaller_batch (torch.Tensor): batch of the same images but cutted. The final shape is (b * sf**2, c, h//sf, w//sf)
  """
  c, h, w = batch.shape[1:]

  if h % sf != 0 or w % sf != 0:
    raise ValueError(f"The shape of the images in input isn't a multiple of the stride_factor! Got {h%sf}, {w%sf}")

  h_sp = h // sf
  w_sp = w // sf

  smaller_batch = batch.unfold(1, c, c).unfold(2, h_sp, h_sp).unfold(3, w_sp, w_sp)
  smaller_batch = smaller_batch.contiguous().view(-1, c, h_sp, w_sp)
  return smaller_batch


def restore_batch(split_batch:torch.Tensor, sf:int):
  """
  Function that receives as input a batch of splitted images of size (b, c, h, w), the shape of the
  original batch before it was splitted and the stride factor used to split it.
  It glues back together the patches to bring back the original images.

    Args:
      - split_batch (torch.Tensor): input batch of size (b, c, h, w)
      - sf (int): stride factor used to split the original batch

    Returns:
      - reconstructed_batch (torch.Tensor): batch where the images has been glued back together
  """
  b, c, h, w = split_batch.shape
  batch_size = b // (sf**2)

  # Calculate the number of subpatches per image
  subpatches_per_image = b // batch_size
  num_images = subpatches_per_image // sf
  channel_split_num = c // c

  # Reshape split batch back to original shape
  split_batch = split_batch.view(batch_size, channel_split_num, num_images, num_images, c, h, w)
  # Reshape to have the same dimensions as original images
  reconstructed_batch = split_batch.permute(0,1,4,2,5,3,6).contiguous().view(batch_size, c, h*sf, w*sf)

  return reconstructed_batch