import glob, os

from ..data_handlers.generic_dataset import GenericDataset

class DNDTrain(GenericDataset):
  """
  Class to be used for DND prepared dataset.
  This dataset is composed of DND images cropped with dimension: (512,512).
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the path to the dataset, checks its existance and calculates the number of images in it.
    """  
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the dataset.\nGot: {self.data_dir}"

    # Read all images in the dataset_path
    self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
    print(f"Found {len(self.full_img_paths)} noisy samples for training.") 
  
  def load_data(self, idx:int):
    """
    This function, given the index of the image to load, finds the correct path, calls  the function that loads it and returns it in a dictionary.
      Args:
        - idx (int): index of the image to load
      Returns:
        - (dict): dictionary containing the image under the key: 'noisy'
    """
    file_path = self.full_img_paths[idx]
    train_img = self.load_image(file_path, as_gray=False)
    return {'noisy': train_img}
  
class DNDValidation(GenericDataset):
  """
  Class to use for the DND validation/test dataset.
  This dataset is composed either of random crops of dimension (256,256) gathered from the images of the training set, or of the official crops indicated by the authors as test set.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def scan_images(self):
    """
    Function that finds the path to the dataset, checks its existance and calculates the number of images in it.
    """
    assert os.path.exists(self.data_dir), f"Couldn't find the path to the dataset.\nGot: {self.data_dir}"

    self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
    print(f'Found {len(self.full_img_paths)} noisy images for validation.')

  def load_data(self, idx:int):
    """
    This function, given the index of the image to load, finds the correct path, calls  the function that loads it and returns it in a dictionary.
      Args:
        - idx (int): index of the image to load
      Returns:
        - (dict): dictionary containing the image under the key: 'noisy'
    """
    file_path = self.full_img_paths[idx]
    noisy_img = self.load_image(file_path, as_gray=False)

    return{'noisy': noisy_img}

if __name__ == '__main__':
  
  # DEBUGGING CODE
  dataset_folder0 = '/mnt/datasets_1/diegom00/dnd_128'
  dataset_folder1 = '/mnt/datasets_1/diegom00/dnd_val'
  dataset_args0 = {'data_dir': dataset_folder0}
  dataset_args1 = {'data_dir': dataset_folder1}

  dataset_train = DNDTrain(**dataset_args0)
  dataset_val = DNDValidation(**dataset_args1)