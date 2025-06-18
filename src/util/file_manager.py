import os
import torch
import numpy as np
import cv2

from ..util.util import tensor_to_np

class FileManager():
  """
  Class used to mnage files and directories.
  """
  def __init__(self, run_name:str, output_folder:str):
    # Create the folder that will contain the output files
    self.output_folder = output_folder
    if not os.path.isdir(self.output_folder):
      os.makedirs(self.output_folder)
      print("The output folder isn't present, generating a new one!")
    
    # Based on the run name, generate a new folder for the results
    self.run_name = run_name
    os.makedirs(os.path.join(self.output_folder, self.run_name), exist_ok=True)

    # Build a separate directory for the checkpoints, the images and the logfiles
    for folder in ['checkpoint', 'best_ckpts', 'tboard']:
      self.make_dir(folder)

  def make_dir(self, dir_name:str):
    """
    Function that creates a directory given the name.
    The directory is created in the output folder!

      Args:
        - dir_name (str): name of the directory to create
    """
    os.makedirs(os.path.join(self.output_folder, self.run_name, dir_name), exist_ok=True)

  def get_dir(self, dir_name:str) -> str:
    """
    Function that given a directory name, returns the path to it in the output folder.

      Args:
        - dir_name (str): name of a directory
      
      Returns:
        - path to the searched directory (str)
    """
    return os.path.join(self.output_folder, self.run_name, dir_name)
 
  def exist_dir(self, dir_name:str) -> bool:
    """
    Function that checks if the directory with the provided name is already present in the output folder.

      Args:
        - dir_name (str): name of a directory
    """
    return os.path.isdir(os.path.join(self.output_folder, self.run_name, dir_name)) 
 
  def save_img_tensor(self, dir_name:str, file_name:str, image:torch.Tensor, extension:str='png'):
    """
    Function that saves an image tensor to a specific directory.
    The tensor is first transformed in an numpy.ndarray that follows the opencv format.
    Be carefull, the function used to save images is OppenCV's imwrite(). Reading from the official 
    docs of the function, it saves only 8-bit unsigned images in the BGR or gray-scale format.

      Args:
        - dir_name (str): name of the directory where the image should be saved
        - file_name (str): name of the file to save
        - image (torch.Tensor): image to save
        - extension (str): extension to apply to the image (default='png')
    """
    file_path = os.path.join(self.get_dir(dir_name), f'{file_name}.{extension}')
    
    img = tensor_to_np(image)

    # Tentative lines that should bring back the normalized values in the range 0,255 from 0,1
    if "_DN" in file_name:
      img = np.clip(img * 255, 0, 255)
    else:  
      img *= 255
    img = img.astype(np.uint8)
    
    if img.shape[2] == 1:
      cv2.imwrite(file_path, np.squeeze(img, 2))
    else:
      cv2.imwrite(file_path, img)
