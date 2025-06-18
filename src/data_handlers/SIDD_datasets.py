import glob
import os
from scipy.io import loadmat
import numpy as np

from ..data_handlers.generic_dataset import GenericDataset

class SIDDPrepTrain(GenericDataset):
  """
  Class to be used for the SIDD prepared dataset.
  This dataset is composed of the SIDD medium images cropped with dimension: (512,512).

  Remember to set the proper path to the folder containing the cropped images!
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the path to the dataset, checks its existance and calculates the number of images in it.
    """
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the training dataset. \n Got: {self.data_dir}"

    # Read all the images in the dataset_path
    self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
    print(f"Found {len(self.full_img_paths)} noisy samples for training.")

  def load_data(self, idx:int):
    """
    This function, given the index of the image to load, finds the correct path, calls the function that loads it and returns it in a disctionary.
      Args:
        - idx (int): index of the image to load
      Returns:
        - dictionary containing the image under the key: 'noisy'
    """
    # retrieve image path
    file_path = self.full_img_paths[idx]

    train_img = self.load_image(file_path, as_gray=False)

    return {'noisy': train_img}

class SIDDValidation(GenericDataset):
  """
  Class to use for the SIDD validation dataset.
  This dataset is composed of the validation set provided by the authors of the SIDD medium. 
  It is read directly from the 2 .mat files:
    - ValidationNoisyBlocksSrgb.mat
    - ValidationGtBlocksSrgb.mat
  
  Remember to set the proper path to the folder containing the 2 files!
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the directory with the dataset, checks its existance, loads the images from the .mat files and computes the number of images found.
    """
    # Find the correct path and check it
    # self.dataset_path = os.path.join(self.data_dir, 'validation')
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the validation dataset.\n Got: {self.data_dir}"

    # find the clean and noisy mat file paths
    clean_mat_file_path = os.path.join(self.data_dir, 'ValidationGtBlocksSrgb.mat')
    noisy_mat_file_path = os.path.join(self.data_dir, 'ValidationNoisyBlocksSrgb.mat')

    # Load the clean and noisy images
    self.clean_images = np.array(loadmat(clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
    self.noisy_images = np.array(loadmat(noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])

    # shape should be something like: (40, 32, 256, 256, 3) -> tot_images = 40x32 = 1280
    assert self.clean_images.shape[:2] == self.noisy_images.shape[:2], f"The clean and noisy images do not match in number!\n Got clean_images: {self.clean_images.shape[:2]}, noisy_images: {self.noisy_images.shape[:2]}"

    # To still provide the __len__ correct functionality, append the right number of empty images to the proper parameter
    for i in range(self.clean_images.shape[0] * self.clean_images.shape[1]):
      self.full_img_paths.append(None)
  
  def load_data(self, idx:int):
    """
    Since images in the 2 .mat files are organized in the following shapes: (40, 32, 256, 256, 3),
    we must convert the index correctly. The index passed will be some number between 0 and __len__ (1280)!
      Args:
        - idx (int): index of the images to extract
      Returns:
        - dictionary with the 2 images found under the keys: 'clean' and 'noisy'
    """
    img_id = idx // 32
    patch_id = idx % 32

    clean_img = self.clean_images[img_id, patch_id, :]
    noisy_img = self.noisy_images[img_id, patch_id, :]

    return {'clean': clean_img, 'noisy': noisy_img} 

class SIDDBenchmark(GenericDataset):
  """
  This class is used for the SIDD Benchmark dataset.
  This dataset is composed of the benchmark set provided by the authors of SIDD.
  It is composed of a matrix that contains the noisy images:
    - BenchmarkNoisyBlocksSrgb.mat
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the directory with the dataset, checks its existance, loads the images from the .mat files and computes the number of images found.
    The dimensions of the images in the .mat file are: (40, 32, 256, 256, 3)
    """
    # Find the correct path and check it
    # self.dataset_path = os.path.join(self.data_dir, 'test')
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the test dataset.\n Got: {self.data_dir}"

    # find the clean and noisy mat file paths
    noisy_mat_file_path = os.path.join(self.data_dir, 'BenchmarkNoisyBlocksSrgb.mat')

    # Load the clean and noisy images
    self.noisy_images = np.array(loadmat(noisy_mat_file_path, appendmat=False)['BenchmarkNoisyBlocksSrgb'])

    # print("Noisy shape: ", self.noisy_images.shape)
    assert len(self.noisy_images.shape) == 5, f"Found data of dimension: {self.noisy_images.shape}"

    # To still provide the __len__ correct functionality, append the right number of empty images to the proper parameter
    for i in range(self.noisy_images.shape[0] * self.noisy_images.shape[1]):
      self.full_img_paths.append(None)
    return

  def load_data(self, idx):
    """
    Since images in the .mat, files are organized in the following shapes: (40, 32, 256, 256, 3), we must convert the index correctly. The index passed will be some number between 0 and __len__ (1280)!
      Args:
        - idx (int): index of the images to extract
      Returns:
        - dictionary with the images found under the key: 'noisy'
    """
    img_id = idx // 32
    patch_id = idx % 32

    noisy_img = self.noisy_images[img_id, patch_id, :]

    return {'noisy': noisy_img} 
