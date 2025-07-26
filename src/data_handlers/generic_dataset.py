import os
import numpy as np
import random
from skimage.io import imread

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TVF
from torchvision.transforms import v2

class GenericDataset(Dataset):
  """
  Basic class for a dataset to be fed to a pytorch DataLoader.
  A more specific dataset should extend this class and implement the scan_images and load_data functions.
    - self.scan_images(self) : function that scans the data paths of all the images and saves the in a list
    - self.load_data(self, idx): function called in the __getitem__() method, it load a single data using the index provided (idx). This data should be returned in the form of a dictionary under the following keys 'clean' for clean images and 'noisy' for noisy ones.
  
    Args:
      - data_dir (str): path to the dataset folder
      - crop_size (list): list of the type [H, W] with 2 integers specifying the size of the image to crop  (default: [250, 250])
      - augmentations (list): list containing the augmentations to perform to the images, for now only 'flip' and 'rotate' are supported. (default: None)
      - n_data (int): number of data to be used. (default: None -> all data in the dataset)
      - repeat_times (int): number of times the data is used. (default: 1)
  """
  def __init__(self, data_dir:str, crop_size:list=[250,250], augmentations:list=None, n_data:int=None, repeat_times:int=1, add_noise:dict=None):
    super(GenericDataset, self).__init__()
    self.data_dir = data_dir
    if not os.path.isdir(self.data_dir):
      raise Exception(f"The provided data directory doesn't exist.\n Got: {self.data_dir}")

    # transform the list's strings into integers
    if crop_size is not None:
      self.cs = list(map(int, crop_size))
    else:
      self.cs = None
    self.aug_list = augmentations
    self.rep_times = int(repeat_times)

    self.full_img_paths = []
    self.scan_images()
    if len(self.full_img_paths) > 0:
      # Check the stored path is actually a class that can be sorted (in validation, Nones are appended to the list to make __len__() funct work)
      if self.full_img_paths[0].__class__.__name__ in ['str', 'int', 'float']:
        self.full_img_paths.sort()

    if n_data is None:
      self.n_data = len(self.full_img_paths)
    else: 
      self.n_data = int(n_data)
    
    self.noise_dict = add_noise

  def scan_images(self):
    # Override the function in the more specific dataset class
    raise NotImplementedError    
  
  def load_data(self, idx):
    # Override the function in the more specific dataset class
    raise NotImplementedError    
 
  def load_image(self, img_path:str, as_gray:bool=False):
    """
    Function that loads an image given the path.
      Args:
        - img_path (str): path to the image to load
        - as_gray (bool): wether to load the image as a gray_scale one (default: False)
    """
    img = imread(img_path, as_gray=as_gray)
    assert img is not None, f"Failure loading the image: {img_path}"
    return img

  def __len__(self):
    return self.n_data * self.rep_times

  def __getitem__(self, idx):
    # Calculate the index since the getitem could be called more than once depending on the value of rep_times
    data_idx = idx % self.n_data

    # load the data
    data = self.load_data(data_idx)

    # crop the image according to the crop_size
    if self.cs is not None:
      data = self.crop_data(data)
    
    # check the type of the cropped image and convert it to a torch tensor
    for key in data:
      if isinstance(data[key], np.ndarray):
        data[key] = torch.from_numpy(np.ascontiguousarray(data[key]))
        # check the shape is of the following kind: (h,w,c). In case convert it to: (c,h,w)
        if data[key].shape[-1] == 3 or data[key].shape[-1] == 1:
          data[key] = data[key].permute(2, 0, 1)
        else:
          raise Exception(f"Weird data shape. Got: {data[key].shape}")

    # augment the data according to the aug_list
    if self.aug_list is not None:
      data = self.augment(data, self.aug_list)

    # Normalize the images in the range [0, 1]
    for key in data:
      data[key] = data[key].type(torch.float32) * 1/255
      # Might augment also noisy images, not sure 100% cause I'm gonna use it only for clean ones
      if self.noise_dict is not None and key == 'noisy':

        photon_scale = int(self.noise_dict['photon_scale'])
        noise_boost = float(self.noise_dict['noise_boost'])
        correlate = self.noise_dict['correlate']
        kernel_edge = int(self.noise_dict['kernel_edge'])
        kernel_sigma = float(self.noise_dict['kernel_sigma'])


        # Sentinel-2 band-specific SNRs (R, G, B = B04, B03, B02)
        snrs = torch.tensor([230, 249, 214], dtype=torch.float32)

        # Add Poisson noise
        poisson = torch.poisson(data[key] * photon_scale) / photon_scale

        # Add Gaussian noise for each band based on the avg pixel value for that band
        avg_per_band = torch.mean(data[key], dim=(1, 2))  # Average across spatial dimensions
        std_dev = avg_per_band / snrs
        
        # Expand std_dev to match image dimensions for broadcasting
        std_dev_expanded = std_dev.view(-1, 1, 1) * noise_boost
        gaussian = torch.normal(0, std_dev_expanded.expand_as(data[key]), dtype=torch.float32)

        if correlate:
          correlated_gaussian = torch.zeros_like(gaussian)
          kernel = self.custom_gkernel(kernel_edge, kernel_sigma)
          for c in range(gaussian.shape[0]):
              correlated_gaussian[c] = F.conv2d(gaussian[c].unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding="same").squeeze(0)
          gaussian = correlated_gaussian

        # Combine noises
        noisy = poisson + gaussian
        
        noisy = torch.clamp(noisy, 0, 1)  # Ensure values are in [0, 1]
        data[key] = noisy
      elif self.noise_dict['correlate'] == 'False': 
        mean = float(self.noise_dict['mean'])
        sigma = float(self.noise_dict['sigma']) / 255
        transform = v2.GaussianNoise(mean=mean, sigma=sigma)
        data[key] = transform(data[key])
      else:
        raise ValueError(f"Weird value for key 'correlate'.\nGot: {self.noise_dict['correlate']}")

    return data
  
  def crop_data(self, data:dict):
    """
    Function that crops the images contained in the data dictionary accordingly to the crop_size provided.
      
      Args:
        - data (dict): dictionary containing at least one of the following keys: 'clean', 'noisy'. Under those, a numpy array representing an image is associated as value. If both are present, the arrays must have the same shape.
      Returns:
        - data (dict): same dictionary as the one in input, but containing the cropped images.
    """ 
    if 'clean' in data and 'noisy' in data:
      assert data['clean'].shape == data['noisy'].shape, f"The clean and noisy images should have \
        the same shapes.\n Got clean_shape: {data['clean'].shape}, noisy_shape: {data['noisy'].shape}"
    
    # Here I use noisy as it should always be contained in data
    if len(data['noisy'].shape) == 3:
      H,W,C = data['noisy'].shape
    elif len(data['noisy'].shape) == 2:
      H,W = data['noisy'].shape
    else:
      raise Exception(f"Weird image shape received. Got: {data['noisy'].shape}")

    rand_h = np.random.randint(0, max(0, H - self.cs[0]))
    rand_w = np.random.randint(0, max(0, W - self.cs[1]))

    for key in data:
      if len(data[key].shape) == 3:
        data[key] = data[key][rand_h:rand_h+self.cs[0], rand_w:rand_w+self.cs[1], :]
      elif len(data[key].shape) == 2:
        data[key] = data[key][rand_h:rand_h+self.cs[0], rand_w:rand_w+self.cs[1]]
        data[key] = data[key][:, :, np.newaxis]
      else:
        raise Exception(f"Unknown image shape. Got: {data[key].shape}")
  
    return data
  
  def augment(self, data:dict, aug_list:list):
    """
    Function used to augment each input image in the batch.
    It is accessed only when the passed aug_list isn't None.

      Args:
        - data (dict): dictionary that contains the images to augment of the type: torch.Tensor.
        - aug_list (list): list of strings of the augmentations to perform
      Returns:
        - data (dict): same dictionay as the one in input, but containing the augmented images.
    """
    for key in data:
      assert isinstance(data[key], torch.Tensor), f"The {key} image in the dictionary is not a torch.Tensor"
      for aug_type in aug_list:
        if aug_type == 'rotate':
          rotation_times = random.randint(0, 3)
          if rotation_times != 0:
            data[key] = torch.rot90(data[key], k=rotation_times, dims=[-2, -1])
        
        elif aug_type == 'flip':
          # flip map:
          # 0: no vflip, no hflip
          # 1: no vflip, ok hflip
          # 2: ok vflip, no hflip
          # 3: ok vflip, ok hflip
          flip_direction = random.randint(0, 3)
          if flip_direction == 0:
            data[key] = data[key]
          elif flip_direction == 1:
            data[key] = TVF.hflip(data[key])
          elif flip_direction == 2:
            data[key] = TVF.vflip(data[key])
          elif flip_direction == 3:
            data[key] = TVF.hflip(data[key])
            data[key] = TVF.vflip(data[key])
          else:
            raise Exception(f"Weird flip direction received. Got: {flip_direction}")
        
        else: 
          raise NotImplementedError(f"The augmentation type provided ({aug_type}) is not implemented yet.")
  
    return data
  
  def custom_gkernel(self, edge:int=3, sigma:float=1.) -> torch.Tensor:
    """
    Custom function that generates a 2D Gaussian kernel.
    
      Args:
        - edge (int): Dimension of the kernel's edge (default=3)
        - sigma (float): Standard deviation of the kernel (default=1.)
    
      Returns:
        - kernel (torch.Tensor): The Gaussian kernel
    """
    # Generate axis values
    ax = torch.linspace(-(edge - 1) / 2., (edge - 1) / 2., edge)

    # Create the 1D Gaussian distribution
    gauss = torch.exp(-0.5 * (ax / sigma).pow(2))

    # Create the 2D Gaussian kernel using the outer product
    kernel = torch.outer(gauss, gauss)

    # Normalize the kernel to ensure it sums to 1
    kernel /= kernel.sum()

    return kernel