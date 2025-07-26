from src.data_handlers.generic_dataset import GenericDataset
import glob
import os
import numpy as np

class Sentinel2Train(GenericDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # load_image should return the noisy version of the image
        train_img = self.load_image(file_path, as_gray=False)

        return { 'noisy': train_img }


class Sentinel2Validation(GenericDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scan_images(self):
        """
        Function that finds the path to the dataset, checks its existence and calculates the number of images in it.
        """
        assert os.path.exists(self.data_dir), f"Couldn't find the path to the validation dataset. \n Got: {self.data_dir}"

        # Read all the images in the dataset_path
        self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
        print(f"Found {len(self.full_img_paths)} clean images for validation.")
  
    def load_data(self, idx:int):
        """
        This function loads a clean image and creates a copy for noise addition.
        The noise will be added later in the GenericDataset.__getitem__() method if add_noise is configured.
        Args:
            - idx (int): index of the image to load
        Returns:
            - dictionary containing both 'clean' and 'noisy' images
        """
        file_path = self.full_img_paths[idx]
        
        # Load the clean image (Sentinel2 is typically multi-channel, so as_gray=False)
        clean_img = self.load_image(file_path, as_gray=False)
        
        # Create a copy for the noisy version - noise will be added automatically by GenericDataset
        noisy_img = np.copy(clean_img)
        
        return { 'clean': clean_img, 'noisy': noisy_img }

class Sentinel2Test(GenericDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def scan_images(self):
        """
        Function that finds the path to the dataset, checks its existence and calculates the number of images in it.
        """
        assert os.path.exists(self.data_dir), f"Couldn't find the path to the test dataset. \n Got: {self.data_dir}"

        # Read all the images in the dataset_path
        self.full_img_paths = glob.glob(os.path.join(self.data_dir, '*'))
        print(f"Found {len(self.full_img_paths)} images for testing.")
  
    def load_data(self, idx:int):
        """
        This function loads a clean image and creates a copy for noise addition (if needed for testing).
        Args:
            - idx (int): index of the image to load
        Returns:
            - dictionary containing both 'clean' and 'noisy' images
        """
        file_path = self.full_img_paths[idx]
        
        # Load the clean image
        clean_img = self.load_image(file_path, as_gray=False)
        
        # Create a copy for the noisy version - noise will be added automatically by GenericDataset if configured
        noisy_img = np.copy(clean_img)
        
        return { 'clean': clean_img, 'noisy': noisy_img } 