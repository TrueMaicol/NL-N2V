from ..util.util import compute_psnr, compute_ssim
from ..util.batch_handlers import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, reshape_batch, restore_batch
from ..loss.losses import Loss
from ..trainer.new_trainer import StdTrainer

# Import torch related things
import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F

class PDTrainer(StdTrainer):
  """
  Trainer class used to train a model with Pixelshuffle Downsampling.
    
    Args:
      - cfg_dict (dict): dictionary containing the information taken from the yaml configuration file.
  """
  def __init__(self, cfg_dict:dict):
    super().__init__(cfg_dict)
  
    # Save the sf value for train and test
    self.sf_tr = int(cfg_dict['model']['kwargs']['pd_training'])
    self.sf_ts = int(cfg_dict['model']['kwargs']['pd_testing'])

    # Set the flag for training with or without padding
    self.val_pad, self.test_pad = self.set_test_val_pad()
  
  # ========================= #
  #     NEW SETTER METHOD     #
  # ========================= #

  def set_test_val_pad(self):
    """
    Setter method for the flag used to pad images in input to remove the color artifacts.
    
      Returns:
        - pad_val (bool): wether to pad or not the validation images
        - pad_test (bool): wether to pad or not the test images
    """
    # Set the validation pad flag
    assert 'val_pad' in self.val_cfg_dict, f"Missing 'val_pad' key!"
    if self.val_cfg_dict['val_pad'] == 'True':
      val_pad = True
    elif self.val_cfg_dict['val_pad'] == 'False':
      val_pad = False
    else:
      raise ValueError(f"Unexpected value in val_pad key.\n Got: {self.val_cfg_dict['val_pad']}")
    
    # Set the test pad flag
    assert 'test_pad' in self.test_cfg_dict, "Missing 'test_pad' key!"
    if self.test_cfg_dict['test_pad'] == 'True':
      test_pad = True
    elif self.test_cfg_dict['test_pad'] == 'False':
      test_pad = False
    else:
      raise ValueError(f"Unexpected value in test_pad key.\n Got: {self.test_cfg_dict['test_pad']}")

    return val_pad, test_pad 

  # =========================== #
  #     OVERWRITTEN METHODS     #
  # =========================== #

  def masking_pipeline(self, data_dict: dict):
    """
    Overwrite the function to introduce th PD.
    It follows these steps:
      - fetches the noisy batch of images to denoise
      - applies PD to remove the noise's spatial correlation
      - splits the PD batch images into their smaller ones, then masks them
      - restores the batch to their original shape, glueing images back together
      - adds the masked downsampled image to the data dictionary
    
      Args:
        - data_dict (dict): dictionary with the batch noisy data to mask
    """
    assert 'noisy' in data_dict, "Couldn't find noisy image in the data dictionary!"

    # Fetch batch to downsample and mask
    batch = data_dict['noisy']

    # Apply PD to the batch
    pd_batch = pixel_shuffle_down_sampling(batch, self.sf_tr)

    # Split the downsampled batch to evenly mask the images
    split_pd_batch = reshape_batch(pd_batch, self.sf_tr)

    # Mask each image and retrieve the applied mask
    masked_spd_batch, mask_spd_batch = self.masker.mask(split_pd_batch, self.sf_tr)

    # Reassemble the batches to the original shape
    masked_pd_batch = restore_batch(masked_spd_batch, self.sf_tr)
    mask_pd_batch = restore_batch(mask_spd_batch, self.sf_tr)

    # Add the masked image to the batch
    data_dict['masked'] = masked_pd_batch
    data_dict['mask'] = mask_pd_batch
  
  def forward_data(self, model:Module, loss:Loss, data:dict):
    """
    Function overwritten to accomodate for the introduction of the PD.
    It extracts the input data, sends it to the network and calculates the loss and temporal information.

      Args:
        - model (nn.Module): pytorch model of the network
        - loss (Loss): loss to be calculated over the input and output of the model
        - data (dict): dictionary containing the input data ('noisy' and/or 'clean' images)
        
      Returns:
        - losses (dict): dictionary that contains the result of the calculated losses
        - tmp_info (dict): dictionary that contains the result of the calculated temporal information
    """
    data_keys = ['noisy', 'masked', 'mask']
    for key in data_keys:
      assert key in data, f"Couldn't find information about key: {key} in the data dictionary!"
    
    # Extract masked and noisy image
    masked_btc = data['masked']
    noisy_btc = data['noisy']

    # Feed the masked image to the network for prediction
    denoised_btc = model(masked_btc)

    # Upsample the denoised image 
    denoised_btc = pixel_shuffle_up_sampling(denoised_btc, self.sf_tr)
    data['mask'] = pixel_shuffle_up_sampling(data['mask'], self.sf_tr)

    # Compute the loss and temporal information
    losses, tmp_info = loss(noisy_btc, denoised_btc, data, model, ratio=(self.epoch-1 + (self.iter-1)/self.tot_data)/self.max_epoch)

    return losses, tmp_info

  def test_or_val_dataloader(self, dataloader:DataLoader, add_constant:int=0, floor:bool=False, img_save_path:str=None, save_image:bool=True, info:bool=True, nos:int=-1, pad_size:int=10):
    """
    Overwritten function to account for the introduction of the PD.

    Function used to test or validate the model over a dataloader's information.
    It follows these steps:
      - Check the existance of the provided directory for saved images
      - Steup the evaluation process
      - Perform the evaluation cycle looping through all the images of the val/test set:
        1. Denoise the image
        2. Compute PSNR and SSIM, if the clean image is provided
        3. Optionally save the produced images
      - Print the final log message to inform about the end of the process
      
      Args:
        - dataloader (torch.Dataloader): iterable dataloader of a specific test/validation dataset
        - add_constant (int): constant to add to the resulting denoised image (default=0)
        - floor (bool): wether to perform the flooring operation over the denoised image (default=False)
        - img_save_path (str): path to the folder where images are saved (default=None)
        - save_image (bool): wether to save images or not (default=True)
        - info (bool): wether to print informations (default=True)
        - nos (int): number of times the images of the validation are saved during a training (default=-1, hence every time)
        - pad_size (int): number of pixels to pad to remove the color artifacts (default=10)
      
      Returns:
        - psnr: total PSNR score of the dataloader results or None (if clean images are not available)
        - ssim: total SSIM score of the dataloader results or None (if clean images are not available)
    """
    # Check if the function has been called during validation or the testing phase
    if img_save_path is not None:
      folder_name = img_save_path.split('/')[-1]
      # Now token will hold the string 'val' or 'test'
      token = folder_name.split('_')[0]
    else:
      raise ValueError("No path has been passed for the folder where images must be saved!")

    psnr_sum = 0.
    ssim_sum = 0.
    count = 0

    for idx, data in enumerate(dataloader):

      # Extract the size of the validation images, since they are all of the same shape do it only at the first iteration
      if idx == 0:
        b,c,h,w = data['noisy'].shape
        # Compute the offset to remove from the image to crop it of the same size of the training ones
        img_edge = int(self.train_cfg_dict['dataset_args']['crop_size'][0])
        offset = (h - img_edge) // 2

      # Send data to the gpu for computation and crop it to the computed offset
      for key in data:
        assert isinstance(data[key], torch.Tensor), f"Unusual input data. Expected Tensor but got:{type(data[key])}"
        data[key] = data[key].cuda()
        # Crop the image to the right size according to the computed offset
        data[key] = data[key][:, :, offset:h-offset, offset:w-offset]
      
      # Add padding to remove color artifacts
      # TODO: experiment with different kinds of paddings
      if token == 'val' and self.val_pad:
        data['noisy'] = F.pad(data['noisy'], (pad_size, pad_size, pad_size, pad_size), "reflect")
      elif token == 'test' and self.test_pad:
        data['noisy'] = F.pad(data['noisy'], (pad_size, pad_size, pad_size, pad_size), "reflect")
      else:
        raise ValueError(f"Token has a weird value.\n Got: {token}")
      
      # Apply PD to data to remove the spatial correlation
      if token == 'val':
        pd_img = pixel_shuffle_down_sampling(data['noisy'], self.sf_tr)
      elif token == 'test':
        pd_img = pixel_shuffle_down_sampling(data['noisy'], self.sf_ts)
      else:
        raise ValueError(f"Token has a weird value.\n Got: {token}")
      
      denoised_img = self.model(pd_img)

      if token == 'val':
        denoised_img = pixel_shuffle_up_sampling(denoised_img, self.sf_tr)
      elif token == 'test':
        denoised_img = pixel_shuffle_up_sampling(denoised_img, self.sf_ts)
      else:
        raise ValueError(f"Token has a weird value.\n Got: {token}")

      # Remove padded pixels
      if self.val_pad:
        denoised_img = denoised_img[:, :, pad_size:-pad_size, pad_size:-pad_size]

      # Add constant and floor
      denoised_img += add_constant
      if floor: denoised_img = torch.floor(denoised_img)

      # Evaluate results
      if 'clean' in data:
        img_psnr = compute_psnr(denoised_img, data['clean'])
        img_ssim = compute_ssim(denoised_img, data['clean'])
        psnr_sum += img_psnr
        ssim_sum += img_ssim
        count += 1
      
      # Compute how many times during training the validation predictions should be saved
      if nos > 0:
        epoch_int = self.max_epoch // nos
      else:
        epoch_int = 1
      
      # Save the validation predictions
      if save_image and (self.epoch % epoch_int) == 0:
        # Check the directory where the images must be saved, if not present create it
        if img_save_path is not None:
          if not self.file_manager.exist_dir(img_save_path):
            self.file_manager.make_dir(img_save_path)
        
        # Send data to the CPU
        if 'clean' in data:
          clean_img = data['clean'].squeeze(0).cpu()
        noisy_img = data['noisy'].squeeze(0).cpu()
        den_img = denoised_img.squeeze(0).cpu()

        # Compute the name of the denoised image
        den_name = f"{idx:04d}_DN_{img_psnr:.2f}" if "clean" in data else f"{idx:04d}_DN"

        # Save images
        if 'clean' in data:
          self.file_manager.save_img_tensor(img_save_path, f"{idx:04d}_CLEAN", clean_img)
        self.file_manager.save_img_tensor(img_save_path, f"{idx:04d}_NOISY", noisy_img)
        self.file_manager.save_img_tensor(img_save_path, den_name, den_img)
      
      # Print information messages
      if info:
        if 'clean' in data:
          self.logger.note(f"[{self.status}] testing... {idx:04d}/{dataloader.__len__():04d}. PSNR: {img_psnr:.2f} dB", end="\r")
        else:
          self.logger.note(f"[{self.status}] testing... {idx:04d}/{dataloader.__len__():04d}", end="\r")
    
    # final log message
    if count > 0:
      self.logger.val(f"[{self.status}] Done! PSNR: {psnr_sum/count:.2f} dB, SSIM: {ssim_sum/count:.3f}")
    else:
      self.logger.val(f"[{self.status}] Done!")
    
    if count != 0:
      return psnr_sum/count, ssim_sum/count
    else:
      return None, None