import glob, os
import numpy as np

from PIL import Image
from ..data_handlers.generic_dataset import GenericDataset

class BSD68Train(GenericDataset):
  """
  Class to be used for the BSD68 dataset provided by N2V authors.
  The dataset is composed of gray images of dimension (180, 180) with an added AWGN of sigma 25.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the path to the dataset, checks its existance and calculates the numnber of images contained.
    NOTE: All contained images are only clean!
    """
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the training dataset. \nGot: {self.data_dir}"

    # Read all the images in the dataset_path
    self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
    print(f"Found {len(self.full_img_paths)} noisy samples for training.")

  def load_data(self, idx:int) -> dict:
    """
    This function, given the index of the image to load, finds it and loads it. Returning such image in a dictionary.
    NOTE: All contained images are only clean!
      Args:
        - idx (int): index of the image to load
      Returns:
        - (dict): dictionary containing the image under the key: 'noisy'
    """
    # retrieve image path
    file_path = self.full_img_paths[idx]

    train_img = self.load_image(file_path, as_gray=True)
    train_img = train_img[:, :, None]

    return {'noisy': train_img}

class BSD68Validation(GenericDataset):
  """
  Class to use for the BSD68 validation dataset.
  This dataset is composed of a signle tiff file of only noisy images.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the path to the dataset, checks its existance and calculates the numnber of images contained.
    NOTE: All contained images are only clean!
    """
    # Check the correctness of the data path
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the validation dataset.\nGot: {self.data_dir}"

    self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
    print(f'Found {len(self.full_img_paths)} noisy images for validation.')

  def load_data(self, idx:int) -> dict:
    """
    This function, given the index of the image to load, finds it and loads it. Returning such image in a dictionary.
    NOTE: All contained images are only clean!
      Args:
        - idx (int): index of the image to load
      Returns:
        - (dict): dictionary containing the image under the key: 'noisy'
    """
    file_path = self.full_img_paths[idx]
    clean_img = self.load_image(file_path, as_gray=True)
    clean_img = clean_img[:, :, None]
    noisy_img = np.copy(clean_img)

    return {'clean': clean_img, 'noisy': noisy_img}
  
class BSD68Test(GenericDataset):
  """
  This class is used for the BSD68 test set.
  The dataset is composed of 68 gray scale images, both on the clean and noisy version.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the path to the dataset, checks its existance and calculates the numnber of images contained.
    NOTE: All contained images are only clean!
    """
    # Check the correctness of the path
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the test dataset.\nGot: {self.data_dir}"

    self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
    print(f"Found {len(self.full_img_paths)} noisy samples for training.")

  def load_data(self, idx:int) -> dict:
    """
    This function, given the index of the image to load, finds it and loads it. Returning such image in a dictionary.
    NOTE: All contained images are only clean!
      Args:
        - idx (int): index of the image to load
      Returns:
        - (dict): dictionary containing the image under the key: 'noisy'
    """
    file_path = self.full_img_paths[idx]
    clean_img = self.load_image(file_path, as_gray=True)
    clean_img = clean_img[:, :, None]
    noisy_img = np.copy(clean_img)

    return {'clean': clean_img, 'noisy': noisy_img}


if __name__ == '__main__':
  
  dataset_folder0 = '/home/diegom00/thesis/datasets/train/Train400'
  dataset_folder1 = '/home/diegom00/thesis/datasets/train/val400'
  dataset_folder2 = '/home/diegom00/thesis/datasets/train/test400'
  dataset_args0 = {'data_dir': dataset_folder0, 'add_noise': {
    'sigma': 25,
    'mean': 0,
    'correlate': True,
    'kernel_sigma': 0.55,
    'kernel_edge': 7
  }}
  dataset_args1 = {'data_dir': dataset_folder1}
  dataset_args2 = {'data_dir': dataset_folder2}

  dataset_train = BSD68Train(**dataset_args0)
  dataset_val = BSD68Validation(**dataset_args1)
  dataset_test = BSD68Test(**dataset_args2)



