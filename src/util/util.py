import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def format_numbers(num):
  """
  Function that given a number formats it by dividing it by 1000 and adding a string symbolizing the magnitude.

    Args:
      - num: Number to format ()int.\:w
      .
    Returns:
      - String of the formatted number.
  """
  magnitude = 0
  mag_list = ['', 'K', 'M', 'G', 'T', 'P']
  while abs(num) >= 1000:
    magnitude += 1
    num /= 1000
  
  return f"{num:.1f}{mag_list[magnitude]}"

def tensor_to_np(img:torch.Tensor):
  """
  Function that converts an image stored in torch.Tensor format to a numpy array.
  It performs a flip of the color-space of the images to convert them from RGB to BGR.
  
    Args:
      - img (torch.Tensor): input image to convert to numpy.ndarray
    
    Returns:
      - (numpy.ndarray): the input image in the opencv format [(RGB) -> (BGR)] [(c,h,w) -> (h,w,c)]
  """
  img = img.cpu().detach()

  # if gray_scale tensor
  if len(img.shape) == 2:
    return img.numpy()
  elif len(img.shape) == 3:
    return np.flip(img.permute(1, 2, 0).numpy(), axis=2)
  elif len(img.shape) == 4:
    return np.flip(img.permute(0, 2, 3, 1).numpy(), axis=3)
  else:
    raise RuntimeError(f"Weird image tensor dimension. Got: {img.shape}")
  
def np_to_tensor(img:np.ndarray):
  """
  Function used to transform a numpy image into a torch tensor. 
  It performs a flip of the color space in case the image is a coloured one. (BGR) -> (RGB)
  It also flips the channels' information: (h,w,c) -> (c,h,w)

    Args:
      - img (np.ndarray): Image to be returned as torch.Tensor
    
    Returns:
      - The input image as a torch.Tensor
  """
  if len(img.shape) == 2:
    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
  elif len(img.shape) == 3:
    return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(img, axis=2), (2, 0, 1))))
  else:
    raise RuntimeError(f"Unexpected dimension of the image to convert. Got: {img.shape}")
  
def compute_psnr(img_den:torch.Tensor, img_clean:torch.Tensor):
  """
  Function that computes the PSNR of the first input image, given the groundtruth.
  Make sure the input images are in the [0,255] data range!

    Args:
      - img_den (torch.Tensor): Denoised image which psnr must be computed
      - img_clean (torch.Tensor): Ground truth image used as comparison for the calculation of the metric 

    Returns:
      - psnr (float): psnr of the 2 images calculated using the skimage.metrics function 
  """
  assert isinstance(img_den, torch.Tensor) and isinstance(img_clean, torch.Tensor), f"Both the images in input to the function should be of the type torch.Tensor.\n Got, img1: {type(img_den)}, img2: {type(img_clean)} ..."

  # remove batch information from both images if present
  if len(img_den.shape) == 4:
    img_den = img_den[0]
  if len(img_clean.shape) == 4:
    img_clean = img_clean[0]
  
  # transform to numpy image
  img_den = tensor_to_np(img_den)
  img_clean = tensor_to_np(img_clean)

  # clip to the [0, 255] range
  # clean image can be safely brought back to the original [0,255] pixel range with a simple multiplication as it isn't touched in the code
  img_clean *= 255
  img_clean = img_clean.astype(np.uint8)
  clipped_img_den = np.clip(img_den * 255, 0, 255)
  clipped_img_den = clipped_img_den.astype(np.uint8)

  return peak_signal_noise_ratio(image_true=img_clean, image_test=clipped_img_den, data_range=255)

def compute_ssim(img_den:torch.Tensor, img_clean:torch.Tensor):
  """
  Function that computes the SSIM of the first input image, given the groundtruth.
  Make sure the input images are in the [0,255] data range!

    Args:
      - img_den (torch.Tensor): Denoised image which psnr must be computed
      - img_clean (torch.Tensor): Ground truth image used as comparison for the calculation of the metric 

    Returns:
      - ssim (float): ssim of the 2 images calculated using the skimage.metrics function 
  """
  assert isinstance(img_den, torch.Tensor) and isinstance(img_clean, torch.Tensor), f"Both the images in input to the function should be of the type torch.Tensor.\n Got, img1: {type(img_den)}, img2: {type(img_clean)} ..."
  
  # remove batch information from both images if present
  if len(img_den.shape) == 4:
    img_den = img_den[0]
  if len(img_clean.shape) == 4:
    img_clean = img_clean[0]
  
  # transform to numpy image
  img_den = tensor_to_np(img_den)
  img_clean = tensor_to_np(img_clean)

  # clip to the [0, 255] range
  # clean image can be safely brought back to the original [0,255] pixel range with a simple multiplication as it isn't touched in the code
  img_clean *= 255
  img_clean = img_clean.astype(np.uint8)
  clipped_img_den = np.clip(img_den * 255, 0, 255)
  clipped_img_den = clipped_img_den.astype(np.uint8)

  assert (img_den.shape[2] == 3 and img_clean.shape[2] == 3) or (img_den.shape[2] == 1 and img_clean.shape[2] == 1), f"Channel axis is not the last one. The shapes of the images are: {img_den.shape}, {img_clean.shape}"

  return structural_similarity(clipped_img_den, img_clean, data_range=255, channel_axis=2) 
