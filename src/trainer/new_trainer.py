import os
import time, datetime
import math
import numpy as np
import cv2
import statistics

from ..util.file_manager import FileManager
from ..util.logger import Logger
from ..util.util import format_numbers, compute_psnr, compute_ssim, np_to_tensor, tensor_to_np
from ..model.denoiser import NL_BSN
from ..data_handlers.SIDD_datasets import SIDDPrepTrain, SIDDValidation, SIDDBenchmark
from ..data_handlers.BSD68_datasets import BSD68Train, BSD68Validation, BSD68Test
from ..data_handlers.DND_datasets import DNDTrain, DNDValidation
from ..loss.losses import Loss
from ..masker.masker import Masker
from ..masker.n2v_masker import N2V_Masker

# Import torch related things
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim

class StdTrainer():
  def __init__(self, cfg_dict:dict):
    # Name of the current run
    self.run_name = cfg_dict['run_name']

    # Name of the folder where checkpoints are stored
    self.checkpoint_folder = 'checkpoint'

    # Build FileManager, class that handles the burden of navigating folder and file names
    self.file_manager = FileManager(cfg_dict['run_name'], cfg_dict['output_folder'])
    # Build Logger, class that handles the process og logging the network's progress
    self.logger = Logger()

    # Setup the dictionaries that keep configuration info
    self.cfg_dict = cfg_dict
    self.train_cfg_dict = cfg_dict['training']
    self.val_cfg_dict = cfg_dict['validation']
    self.test_cfg_dict = cfg_dict['test']
    self.checkpoint_cfg_dict = cfg_dict['checkpoint']

    # Set up the masker
    self.masker = self.set_masker()

  # ========================== #
  #   TRAIN AND TEST METHODS   #
  # ========================== #

  def train(self):
    """
    Function called to train the model over a specific dataset.
    It initializes the training process, than runs each epoch and finally performs the post-processing.
    """

    self.init_training()

    # training
    for self.epoch in range(self.epoch, self.max_epoch+1):
      self.epoch_preprocessing()
      self.epoch_training()
      self.epoch_postprocessing()
    
    self.training_postprocessing()

  @torch.no_grad()
  def test(self):
    """
    Function called to test a model over a specific dataset.
    It can be used to test multiple types of inputs, such as a normal dataset, a directory of images or a single image.
    """
    # If the test has to be performed on a usual dataset -> True, else False
    load_dataset = (self.cfg_dict['test_img'] is None) and (self.cfg_dict['test_dir'] is None)
    self.init_testing(load_dataset)

    # Set the path where images are saved
    for i in range(60):
      test_time = datetime.datetime.now().strftime('%m-%d-%H-%M') + f"-{i:02d}"
      img_save_path = f"img/test_{self.cfg_dict['test']['dataset']}_{self.epoch}_{test_time}"
      # Exit the loop when the new path isn't present
      if not self.file_manager.exist_dir(img_save_path): break
    
    assert 'save_image' in self.test_cfg_dict, f"Couldn't find the 'save_image' key in the test dictionary!"
    # convert string to bool
    if self.test_cfg_dict['save_image'] == 'True':
      save_img = True
    elif self.test_cfg_dict['save_image'] == 'False':
      save_img = False
    else:
      raise ValueError(f"Unusual value for 'save_image' parameter.\n Got: {self.test_cfg_dict['save_image']}")
    
    if 'floor' in self.test_cfg_dict:
      if self.test_cfg_dict['floor'] == 'True':
        floor = True
      elif self.test_cfg_dict['floor'] == 'False':
        floor = False
      else:
        raise ValueError(f"Unusual value for floor parameters.\n Got: {self.test_cfg_dict['floor']}")
    
    # Test over single image
    if self.cfg_dict['test_img'] is not None:
      self.test_img()
    # Test over a directory of images
    elif self.cfg_dict['test_dir'] is not None:
      self.test_dir()
    # Test over a normal dataset
    else:
      psnr, ssim = self.test_or_val_dataloader(
        dataloader=self.test_dataloader,
        add_constant=0. if not 'add_constant' in self.test_cfg_dict else float(self.test_cfg_dict['add_constant']),
        floor=False if not 'floor' in self.test_cfg_dict else floor,
        img_save_path=img_save_path,
        save_image=save_img
      )

      # if the PSNR and SSIM can be computed, print them
      if psnr is not None and ssim is not None:
        with open(os.path.join(self.file_manager.get_dir(img_save_path), f"_average_metrics.txt"), 'w') as f:
          f.write(f"PSNR: {psnr}\nSSIM: {ssim}")

  # ========================= #
  #   TRAINING CORE METHODS   #
  # ========================= #

  def init_training(self):
    """
    Function that initializes the training of NL-BSN.
    It performs the following steps:
      - set the model to be trained (in our case the Unet2)
      - set the dataloaders for both the training and validation datasets (SIDD in our case)
      - set the epochs related parameters
      - set the loss function to be calculated and observed
      - set the optimizer for the model
      - (optional): resume training from a previous checkpoint
      - set logger
      - set tensorboard
      - send model and optimizer to the gpu by calling the .cuda() method of torch
      - print starting messages
    """
    # cudnn
    torch.backends.cudnn.benchmark = False

    # Initialize model
    self.model = self.set_model()
    assert isinstance(self.model, NL_BSN), "The model is incorrectly initialized."

    # Initialize training and validation dataloader
    self.train_data_loader = self.set_dataloader(self.train_cfg_dict, num_workers=self.cfg_dict['thread'])
    self.val_data_loader = self.set_dataloader(self.val_cfg_dict, num_workers=self.cfg_dict['thread'])

    # Initialize epoch parameters
    self.max_epoch = int(self.train_cfg_dict['max_epochs'])
    self.epoch, self.start_epoch = 1, 1
    samples_per_epoch = self.train_data_loader.dataset.__len__()
    self.tot_data = math.ceil(samples_per_epoch/int(self.train_cfg_dict['batch_size']))

    # Initialize loss
    self.loss = Loss(self.train_cfg_dict['loss'], self.train_cfg_dict['tmp_info'])
    self.loss_dict = {'count':0}
    self.tmp_info_dict = {}
    self.loss_log = []

    # Setup the oprimizer
    self.optimizer = self.set_optimizer()
    self.optimizer.zero_grad(set_to_none=True)

    # Resume previous training
    if self.cfg_dict['resume']:
      # find last checkpoint and derive the epoch number
      load_epoch = self.find_last_epoch()

      # Resume the status
      self.set_status(f'epoch {load_epoch:03d}/{self.max_epoch:03d}')

      # Load last checkpoint
      self.load_checkpoint(load_epoch)
      self.epoch = load_epoch + 1

      # Initialize logger, if the training is resumed, the log directory should be already present
      self.logger = Logger(self.max_epoch, self.tot_data, log_dir=self.file_manager.get_dir('logs'), log_file_option='a')
    else:
      # Initialize logger
      self.file_manager.make_dir('logs')
      self.logger = Logger(self.max_epoch, self.tot_data, log_dir=self.file_manager.get_dir('logs'), log_file_option='w')
      # Print the model's information
      self.logger.info(self.summary())

    # Setup tensorboard
    tboard_time = datetime.datetime.now().strftime("%d-%m-%H-%M")
    self.tboard = SummaryWriter(log_dir=self.file_manager.get_dir(f"tboard/{tboard_time}"))

    if self.cfg_dict['gpu'] != 'None':
      # send the model to the GPU
      self.model = nn.DataParallel(self.model).cuda()
      # send the optimizer to the GPU
      for state in self.optimizer.state.values():
        assert isinstance(state, dict), "The optimizer state is not a dictionary!"
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
    else:
      self.model = nn.DataParallel(self.model)

    # Print start message
    self.logger.start(self.epoch - 1, 0)
    self.logger.highlight(self.logger.get_start_msg())
  
  def epoch_preprocessing(self):
    """
    Function called at the beginning of an epoch's training process.
    It follows these steps:
      - set the status (string of current epoch over the entire num of epochs)
      - makes the dataloader iterable
      - set the model in training mode, calling torch's train() function
    """
    # Status is the strin of the current epoch over all the possible ones
    self.set_status(f"epoch {self.epoch:03}/{self.max_epoch:03}")

    # make the dataloader iterable
    self.train_data_loader_iter = iter(self.train_data_loader)

    # set the model in training mode
    self.model.train()
  
  def epoch_training(self):
    """
    Function that runs the epoch, iterating over the entire training dataset.
    """
    for self.iter in range(1, self.tot_data+1):
      self.run_step()
      self.post_step()
  
  def run_step(self):
    """
    Function that runs the core part of each epoch.
    It performs the following steps:
      - fetch data from the trainind dataloader
      - send data to the gpu
      - mask the input data calling the masker
      - compute forward step (calculating the loss)
      - compute backward step
      - perform optimizer step
      - save loss and temporal information in the proper dictionaries
      """
    # Fetch data from the dataloader, data is a dictionary with images
    data = next(self.train_data_loader_iter)

    # Send data to the proper device
    if self.cfg_dict['gpu'] != 'None':
      for key in data:
        assert isinstance(data[key], torch.Tensor), f"Unusual data type returned by the training dataloader.\n Got: {type(data[key])}."
        data[key] = data[key].cuda()
    else:
      for key in data:
        data[key] = data[key].cpu()
    
    # print("In mask: ", time.time())
    # mask the input data with the masker
    self.masking_pipeline(data)
    # print("Out mask: ", time.time())

    # forward data to the model and calculate the loss
    losses, tmp_info = self.forward_data(self.model, self.loss, data)
    losses = {key: losses[key].mean() for key in losses}
    tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

    # Compute the backward step
    tot_loss = sum(v for v in losses.values())
    tot_loss.backward()

    # optimizer step
    self.optimizer.step()
    # zero grad
    self.optimizer.zero_grad(set_to_none=True)

    # save losses and temporal information
    for key in losses:
      if key != 'count':
        if key in self.loss_dict:
          self.loss_dict[key] += float(losses[key])
        else:
          self.loss_dict[key] = float(losses[key])
    
    for key in tmp_info:
      if key in self.tmp_info_dict:
        self.tmp_info_dict[key] += float(tmp_info[key])
      else:
        self.tmp_info_dict[key] = float(tmp_info)
    
    self.loss_dict['count'] += 1
  
  def post_step(self):
    """
    Function called each epoch after performing one step.
    It performs the following operations:
      - adjust the learning rate of the model
      - prints the loss calculated at the current step
      - prints the overall progress
    """
    self.adjust_learning_rate()

    # Print loss every interval/end of epoch
    if (self.iter % int(self.cfg_dict['log']['iter_interval']) == 0) or (self.iter == self.tot_data):
      self.print_loss()
    
    # NB: epochs and iters start from 1 but the progress is computed from 0.
    self.logger.print_log_msg(self.epoch-1, self.iter-1)  
  
  def epoch_postprocessing(self):
    """
    Function called at the end of the epoch's training process.
    It follows these steps:
      - saves a checkpoint of the training (depending on the current epoch and relative configuration's parameters)
      - starts the validation process
    """
    dict_keys = ['start_epoch', 'interval_epoch', 'save_all']
    
    # Check all the necessary information have been provided
    for key in dict_keys:
      assert key in self.checkpoint_cfg_dict, f"Couldn't find the information about {key} among the checkpoint dictionary!"
    if self.checkpoint_cfg_dict['save_all'] == 'True':
      save_all = True
    else:
      save_all = False
    
    # save a checkpoint of the training process
    ckpt_start_epoch = int(self.checkpoint_cfg_dict['start_epoch'])
    ckpt_interval_epoch = int(self.checkpoint_cfg_dict['interval_epoch'])

    if self.epoch >= ckpt_start_epoch:
      # check if the epoch is one during which the network must be checkpointed
      if (self.epoch - ckpt_start_epoch) % ckpt_interval_epoch == 0:
        self.save_checkpoint(save_all)
    
    # start the validation process
    dict_keys.remove('save_all')
    dict_keys.append('val')
    for key in dict_keys:
      assert key in self.val_cfg_dict, f"Couldn't find information about {key} in the validation dictionary!"
    
    val_start_epoch = int(self.val_cfg_dict['start_epoch'])
    val_interval_epoch = int(self.val_cfg_dict['interval_epoch'])
  
    if self.val_cfg_dict['val'] == 'True':
      if (self.epoch - val_start_epoch) % val_interval_epoch == 0:
        self.model.eval()
        self.set_status(f"val {self.epoch:03d}")
        self.validation()
  
  def training_postprocessing(self):
    """
    Function that finalizes the training process by printing a final message.
    """
    self.logger.highlight(self.logger.get_end_msg())

  @torch.no_grad()
  def validation(self):
    """
    Function that performs the validation step at the end of an epoch.
    It does the following operations:
      - sets a directory where the validation images are saved
      - calls the method that goes throigh the validation set and computes the PSNRs and SSIMs of the images
      - saves the best weights for both PSNR and SSIM
    """
    # make a directory for saving images
    img_save_path = f"img/val_{self.epoch:03d}"
    
    # check the presence of the some flags
    assert 'save_image' in self.val_cfg_dict, f"Couldn't find the save_image information in the validation dictionary!"
    if self.val_cfg_dict['save_image'] == 'True':
      save_img = True
    elif self.val_cfg_dict['save_image'] == 'False':
      save_img = False
    else:
      raise ValueError(f"Unsusual value for save_image parameter.\n Got: {self.val_cfg_dict['save_image']}")

    assert 'num_of_saves' in self.val_cfg_dict, f"Couldn't find information about the num_of_saves in the validation dictionary!"
    num_of_saves = int(self.val_cfg_dict['num_of_saves'])

    if 'floor' in self.val_cfg_dict:
      if self.val_cfg_dict['floor'] == 'True':
        floor = True
      elif self.val_cfg_dict['floor'] == 'False':
        floor = False
      else: 
        raise ValueError(f"Unusual value for floor parameter.\n Got {self.val_cfg_dict['floor']}")

    psnr, ssim = self.test_or_val_dataloader(
      dataloader=self.val_data_loader,
      add_constant=0. if not 'add_constant' in self.val_cfg_dict else float(self.val_cfg_dict['add_constant']),
      floor=False if not 'floor' in self.val_cfg_dict else floor,
      img_save_path=img_save_path,
      save_image=save_img,
      nos=num_of_saves
    )

    # At the end of the validation set, given the PSNR and SSIM, save the best weights
    if psnr != None and ssim != None:
      self.save_best_checkpoint(psnr, ssim)

  def test_or_val_dataloader(self, dataloader:DataLoader, add_constant:int=0, floor:bool=False, img_save_path:str=None, save_image:bool=True, info:bool=True, nos:int=-1):
    """
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
      
      Returns:
        - psnr: total PSNR score of the dataloader results or None (if clean images are not available)
        - ssim: total SSIM score of the dataloader results or None (if clean images are not available)
    """
    psnr_sum = 0.
    ssim_sum = 0.
    count = 0
    
    all_psnr = []
    all_ssim = []

    for idx, data in enumerate(dataloader):

      # Extract the size of the validation images, since they are all of the same shape do it only at the first iteration
      if idx == 0 and self.val_cfg_dict['dataset_args']['crop_size'] is not None:
        b,c,h,w = data['noisy'].shape
        # Compute offset to remove from the image to crop it of the same size of the training ones
        img_edge = int(self.train_cfg_dict['dataset_args']['crop_size'][0])
        offset = (h - img_edge) // 2
      else:
        b,c,h,w = data['noisy'].shape
        offset = 0
      
      # Send data to the gpu for computation and crop it to the right dimension
      for key in data:
        assert isinstance(data[key], torch.Tensor), f"Unusual input data. Expected Tensor but got: {type(data[key])}"
        data[key] = data[key].cuda()
        # Cut the image to the right size according to the computed offset
        data[key] = data[key][:, :, offset:h-offset, offset:w-offset]
      
      # Send data to the model to produce the prediction, no need to mask at validation/test time!
      denoised_img = self.model(data['noisy'])

      # Add constant and floor
      denoised_img += add_constant
      if floor: denoised_img = torch.floor(denoised_img)
    
      # Evaluate the metrics
      if 'clean' in data:
        img_psnr = compute_psnr(denoised_img, data['clean'])
        img_ssim = compute_ssim(denoised_img, data['clean'])
        psnr_sum += img_psnr
        ssim_sum += img_ssim
        all_psnr.append(img_psnr)
        all_ssim.append(img_ssim)
        count += 1
      
      # Compute how many times during training the validation predictions should be saved
      if nos > 0:
        epoch_int = self.max_epoch // nos
      else:
        epoch_int = 1
      
      # Save the predictions of the validation set
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
    
      # Save all the metrics in a file if we are testing
      if img_save_path.split('/')[-1].split('_')[0] == 'test' and 'clean' in data:
        with open(os.path.join(self.file_manager.get_dir(img_save_path), f"_all_metrics.txt"), 'a') as f:
          f.write(f"Image {idx:04d}\tPSNR: {img_psnr:.5f}\tSSIM: {img_ssim:.5f}\n")
      
    # final log message
    if count > 0:
      self.logger.val(f"[{self.status}] Done! PSNR: {psnr_sum/count:.2f} dB, SSIM: {ssim_sum/count:.3f}")
    else:
      self.logger.val(f"[{self.status}] Done!")
    
    if count != 0:
      psnr_median = statistics.median(all_psnr)
      ssim_median = statistics.median(all_ssim)
      print(f"\nPSNR MEDIAN: {psnr_median:.2f}\nSSIM MEDIAN: {ssim_median:.3f}")
      return psnr_sum/count, ssim_sum/count
    else:
      return None, None

  # ======================= #
  #     TESTING METHODS     #
  # ======================= #

  def init_testing(self, load_dataset:bool):
    """
    Function that initializes the testing process.
    
      Args: 
      - load_dataset (bool): wether to load a dataset as for training or not
    """
    # init model and status string
    self.model = self.set_model()
    self.status = self.set_status('test')

    # Load checkpoint files
    ckpt_epoch = self.find_last_epoch() if self.cfg_dict['ckpt_epoch'] == -1 else self.cfg_dict['ckpt_epoch']
    ckpt_name = self.cfg_dict['pretrained'] if self.cfg_dict['pretrained'] is not None else None

    self.load_checkpoint(ckpt_epoch, ckpt_name)
    self.epoch = self.cfg_dict['ckpt_epoch']

    # load the dataset
    if load_dataset:
      self.test_dataloader = self.set_dataloader(self.test_cfg_dict, num_workers=self.cfg_dict['thread'])
    
    # Send model to the GPU
    if self.cfg_dict['gpu'] != 'None':
      self.model = nn.DataParallel(self.model).cuda()
    else:
      self.model = nn.DataParallel(self.model)
    
    # Set the model in evaluation mode
    self.model.eval()
    self.set_status(f"test {self.epoch:03d}")

    # produce a start message
    self.logger.highlight(self.logger.get_start_msg())

  def test_img(self, img_path:str, save_dir:str="./", add_constant:float=0., floor:bool=False):
    """
    Function used to perform the inference of the network over a single image.
      Args:
        - img_path (str): path of the image to be denoised 
        - save_dir (str): path of the directory where the denoised image should be saved (default="./)
        - add_constant (float): constant to add for regularization (default=0.)
        - floor (bool): wether to floor the values or not (default=False)
    """
    # Read input
    input_img = np_to_tensor(cv2.imread(img_path))
    # Add batch channel and bring the values to float
    input_img = input_img.unsqueeze(0).float()

    # Normalize
    input_img = input_img / 255

    # Send data to the GPU
    if self.cfg_dict['gpu'] != 'None':
      input_img = input_img.cuda()

    # Perform cropping to bring the test image to the training dimension
    b,c,h,w = input_img.shape
    img_edge = int(self.train_cfg_dict['dataset_args']['crop_size'][0])
    # if the image is bigger than the training image
    if h >= img_edge:
      offset = (h - img_edge) // 2
      input_img = input_img[:, :, offset:h-offset, offset:w-offset]
    # if image is smaller than the training image
    else:
      offset = (img_edge - h) // 2
      if (img_edge % 2) == 0:
        input_img = F.pad(input_img, (offset, offset, offset, offset), "reflect")
      else:
        # if odd, adjust the padding so to have the correct dimension
        input_img = F.pad(input_img, (offset, offset+1, offset, offset+1), "reflect")

    # add constant and round values, if needed
    denoised_img += add_constant
    if floor: denoised_img = torch.floor(denoised_img)

    # save image
    denoised_img = tensor_to_np(denoised_img)
    # remove batch info
    denoised_img = denoised_img.squeeze(0)
    # bring back to uint8 format
    denoised_img *= 255
    denoised_img = denoised_img.astype(np.uint8)

    name = img_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(save_dir, name + '_DN.png'), denoised_img)

    # print log message
    self.logger.note(f"[{self.status}] saved: {os.path.join(save_dir, name + '_DN.png')}")
  
  def test_dir(self, dir_path:str):
    """
    Function that performs inference oevr all the images in the provided directory.
    Be carefull, the function doesn't check the extensions of the files in the directory. Each file is therefore treated like an image would.
    
      Args:
        - dir_path (str): path to the directory containing the images to denoise
    """
    assert os.path.isdir(dir_path), f"Couldn't find the specified image directory.\n Got: {dir_path}"

    # retrieve a list of all the names of the images in the directory
    all_img_paths = [path for path in os.listdir(dir_path) if os.path.isfile(dir_path, path)]
    for img_path in all_img_paths:
      os.makedirs(os.path.join(dir_path, 'results'), exist_ok=True)
      self.test_img(os.path.join(dir_path, img_path), os.path.join(dir_path, 'results'))

  # ====================== #
  #     SETTER METHODS     #
  # ====================== #

  def set_masker(self):
    """
    Setter method for the masker used during pre-processing.

      Returns:
        - masker: The masker to be used during pre-processing
    """
    assert 'masker_type' in self.cfg_dict['masker'], f"Unspecified masker type! Got keys: {self.cfg_dict.keys()}"
    # Get masker type
    masker_type = self.cfg_dict['masker']['masker_type']
    
    assert 'masker_args' in self.cfg_dict['masker'], f"Unspecified masker arguments! Got keys: {self.cfg_dict.keys()}"
    # Check which kind of masker is to be used
    if masker_type == 'Masker':
      masker = Masker(self.cfg_dict['masker']['masker_args'])
    elif masker_type == 'N2V_Masker':
      masker = N2V_Masker(self.cfg_dict['masker']['masker_args'])
    else:
      raise RuntimeError(f"Weird type of masker provided. Got: {masker_type}")
  
    return masker

  def set_model(self) -> nn.Module:
    """
    Function that set the model used for denoising.
    
      Returns:
        - model (nn.Module): The model initialized according to the configuration file, if provided.
    """
    # If the setting is specified, use it otherwise use the default values
    if self.cfg_dict['model']['kwargs'] is not None:
      model = NL_BSN(**self.cfg_dict['model']['kwargs'])
    else:
      model = NL_BSN()
    
    return model

  def set_dataloader(self, dataset_cfg_dict:dict, num_workers:int) -> DataLoader:
    """
    Function that given the configuration dictionary of the dataset, sets the dataset class and returns the torch DataLoader of the given dataset.
    
      Args:
        - dataset_cfg_dict (dict): dictionary contining the info necessary to setup the dataset class.
        - num_workers (int): number of workers to set in the DataLoader.
        
      Returns:
        - dataloader (torch.DataLoader): loader for the provided dataset.
    """
    dataset_name = dataset_cfg_dict['dataset']
    # assert dataset_name in ['SIDD_train', 'SIDD_val', 'SIDD_test', 'BSD68_train', 'BSD68_val', 'BSD68_test'], f"Unknown dataset name!\n Got: {dataset_name}"

    # Check for batch_size
    assert "batch_size" in dataset_cfg_dict, "Batch size not provided in the configuration file!"
    btc_size = dataset_cfg_dict['batch_size']

    # Check for dataset arguments
    assert "dataset_args" in dataset_cfg_dict, "Dataset arguments not provided!"
    dataset_args = dataset_cfg_dict['dataset_args']
    assert isinstance(dataset_args, dict), f"Dataset arguments have been provided with the wrong format!\n Should be a dict, got: {type(dataset_args)}"

    if dataset_name == 'SIDD_train':
      dataset = SIDDPrepTrain(**dataset_args)
      shuffle = True 
    elif dataset_name == 'SIDD_val':
      dataset = SIDDValidation(**dataset_args)
      shuffle = False
    elif dataset_name == 'SIDD_test':
      dataset = SIDDBenchmark(**dataset_args)
      shuffle = False
    elif dataset_name == 'BSD68_train':
      dataset = BSD68Train(**dataset_args)
      shuffle = False
    elif dataset_name == 'BSD68_val':
      dataset = BSD68Validation(**dataset_args)
      shuffle = False
    elif dataset_name == 'BSD68_test':
      dataset = BSD68Test(**dataset_args)
      shuffle = False
    elif dataset_name == 'DND_train':
      dataset = DNDTrain(**dataset_args)
      shuffle = True
    elif dataset_name == 'DND_val':
      dataset = DNDValidation(**dataset_args)
      shuffle = False
    else:
      raise NotImplementedError(f"Class for dataset: {dataset_name}, has not been implemented yet.")

    # pin_memory = True
    dataloader = DataLoader(
      dataset=dataset,
      num_workers=num_workers,
      batch_size=int(btc_size),
      shuffle=shuffle,
      pin_memory=True
    )   

    return dataloader

  def set_optimizer(self):
    """
    Setter function for the network's optimizer.
    
      Returns:
        - torch.optim: the optimizer to use for the network
    """
    opt = self.train_cfg_dict['optimizer']
    parameters = self.model.parameters()
    lr = float(self.train_cfg_dict['init_lr'])

    assert 'type' in opt, "Couldn't find the type of the optimizer in the configuration dictionary."
    if opt['type'] == 'Adam':
      # convert elements of the list from str to floats
      betas = list(map(float, opt['Adam']['betas']))
      return optim.Adam(parameters, lr=lr, betas=betas)
    else:
      raise NotImplementedError(f"The provided optimizer type ({opt['type']}) has not been implemented yet!")
  
  def set_status(self, status:str):
    """
    Setter function for the status string, basically a string that shows the state of the training or validation process in terms of epochs.
    
      Args:
        - status (str): string to modify, then display.
    """
    stat_len = 15
    assert len(status) <= stat_len, f"Status string cannot exceed {stat_len} characters.\n Got: {len(status)}."

    if len(status.split(" ")) == 2:
      stat0, stat1 = status.split(" ")
      self.status = f"{stat0.rjust(stat_len//2)}" + f" {stat1.ljust(stat_len//2)}"
    else:
      space = stat_len - len(status)
      self.status = "".ljust(space//2) + status + "".ljust(space//2)

  # ====================== #
  #     UTILITY METHODS    #
  # ====================== #

  def masking_pipeline(self, data_dict:dict):
    """
    Function that masks the input.
    It calls the masker that performs the substitution and generates the mask.
    
      Args:
        - data_dict (dict): dictionary with the batch noisy data to mask
    """
    assert "noisy" in data_dict, "Couldn't find the noisy batch of images!"

    # Extract batch
    batch = data_dict['noisy']

    # Send batch to masker to obtain masked images and masks
    masked_batch, mask_batch = self.masker.mask(batch)

    # Add the newly produced images to the dictionary
    data_dict['masked'] = masked_batch
    data_dict['mask'] = mask_batch

  def forward_data(self, model:nn.Module, loss:Loss, data:dict):
    """
    Function that extracts the input data, sends it tp the network and calculates the loss and temporal information.

      Args: 
        - model (nn.Module): model of the network
        - loss (Loss): loss to be calculated over the input and output of the model
        - data (dict): dictionary containing the input data (noisy and/or clean images)
      
      Returns:
        - losses (dict): dictionary that contains the result of the calculated loss
        - tmp_info (dict): dictionary that contains the result of the calculated temporal information
    """
    data_keys = ['noisy', 'masked', 'mask']
    for key in data_keys:
      assert key in data, f"Couldn't find {key} image among the ones contained in the dictionary!"
    
    # Extract masked and noisy image
    masked_btc = data['masked']
    noisy_btc = data['noisy']

    # Feed the masked image to the network for prediction
    denoised_btc = model(masked_btc)
    # Send data to the loss class for its proper computation
    losses, tmp_info = loss(noisy_btc, denoised_btc, data, model, ratio=(self.epoch-1 + (self.iter-1)/self.tot_data)/self.max_epoch)
    
    # DEBUG: Compute the time it takes for the network to complete an epochlearning the identity
    # losses, tmp_info = loss(noisy_btc, denoised_btc, data, model, ratio=(self.epoch-1 + (self.iter-1)/self.tot_data)/self.max_epoch)

    return losses, tmp_info
  
  def adjust_learning_rate(self):
    """
    Function thst checks the type of scheduler provided in the configuration file and adjust the learning rate accordingly each epoch.
    """
    scheduler = self.train_cfg_dict['scheduler']
    assert 'type' in scheduler, f"The type for the scheduler has not been provided in the configuration file.\n Got: {scheduler}"

    if scheduler['type'] == 'step':
      # Step decreasing scheduler
      if self.iter == self.tot_data:
        step_sched_dict = scheduler['step']
        assert 'step_size' in step_sched_dict, "Couldn't find the step_size key in the step scheduler"
        assert 'gamma' in step_sched_dict, "Couldn't find the gamma key in the step scheduler"
        if self.epoch % int(step_sched_dict['step_size']) == 0:
          lr_before = self.optimizer.param_groups[0]['lr']
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_before * float(step_sched_dict['gamma'])
    else:
      raise NotImplementedError(f"The provided scheduler type ({scheduler['type']}) has not been implemented yet!")
  
  def get_curr_lr(self):
    """
    Getter function for the current learning rate in use.
    """
    for param_group in self.optimizer.param_groups:
      return param_group['lr']

  # ============================= #
  #     CHECKPOINTING METHODS     #
  # ============================= #

  def find_last_epoch(self) -> int:
    """
    Function that finds the last epoch of the previous training for the given session.
    It searches in the checkpoint folder, retrieves the epochs which  weights have been saved and returns the greatest one.

      Returns:
        - (int): last epoch saved
    """
    chkpt_list = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))

    # remove str info that is not related to the epoch and retrieve epoch number
    epochs = [int(ckpt.replace(f"{self.run_name}_", "").replace(".pth", "")) for ckpt in chkpt_list]
    assert len(epochs) > 0, f"There is no resumable checkpoint on session: {self.run_name}"
    return max(epochs)
  
  def load_checkpoint(self, load_epoch:int=0, name:str=None):
    """
    Function that loads a checkpoint from the proper folder.
    It follows these steps:
      - Builds the file_name (handling the case the name provided is None)
      - Asserts the existance of the file
      - Loads the checkpoint's information calling torch.load()
      - Sets the proper parameter with the gathered information
      - Prints an output message

      Args:
        - load_epoch (int): epoch which checkpoint's belong to (default=0)
        - name (str): name of the checkpoint's file (default=None)
    """
    # If the name is unknown, build it and load from the specific folder
    if name is None:
      if load_epoch == 0:
        return
      file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), f"{self.run_name}_{load_epoch:03d}.pth")
    else:
      # Go through the correct folder and pick up the checkpoint. Since the folders are 2, find the one that contains the given checkpoint
      ckpt_fold = self.file_manager.get_dir(dir_name='checkpoint')
      best_ckpt_fold = self.file_manager.get_dir(dir_name='best_ckpts')
      if name in os.listdir(ckpt_fold):
        file_name = os.path.join(ckpt_fold, name)
      elif name in os.listdir(best_ckpt_fold):
        file_name = os.path.join(best_ckpt_fold, name)
      else:
        raise RuntimeError(f"Couldn't find the given checkpoint in the proper folders.\n Got {name}.")
    
    # check the file's existance
    assert os.path.isfile(file_name), f"Couldn't find the checkpoint's file.\n Got: {file_name}"

    # Load the checkpoint
    saved_ckpt = torch.load(file_name)
    # checkpoint is a dictionary with the following keys: ['epoch', 'model_weights', 'optimizer_weights']
    self.epoch = saved_ckpt['epoch']
    #generate a new dictionary to remove the "module." string from the keys of the model_weights
    mod_weights = {}
    for k,v in saved_ckpt['model_weights'].items():
      if k.startswith("module."):
        k = k.replace("module.", "")
      mod_weights[k] = v
    
    self.model.load_state_dict(mod_weights)
    if hasattr(self, "optimizer"):
      self.optimizer.load_state_dict(saved_ckpt['optimizer_weights'])
    
    self.logger.note(f"[{self.status}] model loaded: {file_name}")
  
  def save_checkpoint(self, _all:bool=True):
    """
    Function that computes the name of the current checkpoint and saves it to the checkpoint folder.
    For each checkpoint a dictionary is saved containing:
      - epoch: current epoch at the time of the save
      - model_weights: weights of the model at the time of the checkpoint
      - optimizer_weights: weights of the optimizer at the time of the checkpoint
      
      Args:
        - _all (bool): wether to save all checkpoints or only the last 2 (default=True)
    """
    # create checkpoint name
    ckpt_name = self.run_name + f"_{self.epoch:03d}.pth"
    if _all:
      torch.save({
        'epoch': self.epoch,
        'model_weights': self.model.state_dict(),
        'optimizer_weights': self.optimizer.state_dict()
      }, os.path.join(self.file_manager.get_dir(self.checkpoint_folder), ckpt_name))
    else:
      # Save the last 2 checkpoints
      if len(os.listdir(self.file_manager.get_dir(self.checkpoint_folder))) == 2:
        # get a list of the chekpoints, sort it and remove the first one (oldest)
        list_ckpts = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))
        list_ckpts.sort()
        to_delete = list_ckpts[0]
        os.remove(os.path.join(self.file_manager.get_dir(self.checkpoint_folder), to_delete))
      # Save the checkpoint
      torch.save({
        'epoch': self.epoch,
        'model_weights': self.model.state_dict(),
        'optimizer_weights': self.optimizer.state_dict()
      }, os.path.join(self.file_manager.get_dir(self.checkpoint_folder), ckpt_name))

  def save_best_checkpoint(self, psnr:float, ssim:float):
    """
    Function that saves the checkpoint of the epoch that performed best in terms of PSNR and/or SSIM
    
      Args:
        - psnr (float): average value of the computed psnr on the validation set
        - ssim (float): average value of the computed ssim on the validation set
    """
    best_ckpt_dir = self.file_manager.get_dir('best_ckpts')
    psnr_ckpt_name = self.run_name + f"_{self.epoch:03d}_psnr_{psnr:.2f}.pth"
    ssim_ckpt_name = self.run_name + f"_{self.epoch:03d}_ssim_{ssim:.3f}.pth"

    if len(os.listdir(best_ckpt_dir)) != 0:
      # Get a list of the names of the files in the folder
      file_names = os.listdir(best_ckpt_dir)
      # Remove useless info and retrieve old psnr and ssim
      temp_names = [f_name.replace(f"{self.run_name}_", "").replace(".pth", "") for f_name in file_names]
      for f_name in temp_names:
        if 'psnr' in f_name:
          old_psnr = float(f_name.split('_')[-1])
        elif 'ssim' in f_name:
          old_ssim = float(f_name.split('_')[-1])
        else:
          raise RuntimeError("Couldn't find old PSNR and SSIM!")
    
      # Confront the new and old values, then save the checkpoint is the new one is better
      if psnr > old_psnr:
        # Find the old checkpoint and remove it
        old_psnr_ckpt = file_names[0] if 'psnr' in file_names[0] else file_names[-1]
        os.remove(os.path.join(best_ckpt_dir, old_psnr_ckpt))
        # Save the new one
        torch.save({
          'epoch': self.epoch,
          'model_weights': self.model.state_dict(),
          'optimizer_weights': self.optimizer.state_dict()
        }, os.path.join(best_ckpt_dir, psnr_ckpt_name))

      if ssim > old_ssim:
        # Find the old checkpoint and remove it
        old_ssim_ckpt = file_names[0] if 'ssim' in file_names[0] else file_names[-1]
        os.remove(os.path.join(best_ckpt_dir, old_ssim_ckpt))
        # Save the new one
        torch.save({
          'epoch': self.epoch,
          'model_weights': self.model.state_dict(),
          'optimizer_weights': self.optimizer.state_dict()
        }, os.path.join(best_ckpt_dir, ssim_ckpt_name))
    else:
      # Save first checkpoint for PSNR and SSIM
      torch.save({
        'epoch': self.epoch,
        'model_weights': self.model.state_dict(),
        'optimizer_weights': self.optimizer.state_dict()
      }, os.path.join(best_ckpt_dir, psnr_ckpt_name))
      torch.save({
        'epoch': self.epoch,
        'model_weights': self.model.state_dict(),
        'optimizer_weights': self.optimizer.state_dict()
      }, os.path.join(best_ckpt_dir, ssim_ckpt_name))

  # ====================== #
  #     WRITER METHODS     #
  # ====================== #

  def summary(self):
    """
    Function that builds a summary string of the current model in use.
    
      Returns:
        - summary (str): string containing the information.
    """
    summary = ""
    summary += "-"*100 + "\n"

    # model summary
    param_num = sum(p.numel() for p in self.model.parameters())

    summary += f"Model parameters: {format_numbers(param_num)}"
    summary += str(self.model) + "\n\n"

    summary += "-"*100 + "\n"
    return summary

  def print_loss(self):
    """
    Function called every 'iter_interval' times and at the end of each epoch. It prints the losses calculated at the time is called.
    """
    temp_loss = 0.
    for key in self.loss_dict:
      if key != 'count':
        temp_loss += self.loss_dict[key] / self.loss_dict['count']
    self.loss_log += [temp_loss]
    # do not exceed the 100 elements
    if len(self.loss_log) > 100: self.loss_log.pop(0)

    # Print status and learning rate
    loss_out = f"[{self.status}] {self.iter:04d}/{self.tot_data:04d}, lr: {self.get_curr_lr():.1e} | "
    overall_iter = (self.epoch-1) * self.tot_data + self.iter

    # Print the losses
    avg_loss = np.mean(self.loss_log)
    loss_out += f"avg_100: {avg_loss:.5f} | "
    self.tboard.add_scalar('loss/avg_100', avg_loss, overall_iter)

    for key in self.loss_dict:
      if key != 'count':
        loss = self.loss_dict[key]/self.loss_dict['count']
        loss_out += f"{key}: {loss:.5f}"
        self.tboard.add_scalar(f'loss/{key}', loss, overall_iter)
        self.loss_dict[key] = 0.
    
    # Print temporal info
    if len(self.tmp_info_dict) > 0:
      loss_out += "\t["
      for key in self.tmp_info_dict:
        loss_out += f" {key}: {self.tmp_info_dict[key]/self.loss_dict['count']:.2f}"
        self.tmp_info_dict[key] = 0.
      loss_out += "]"
    
    # reset the counter
    self.loss_dict['count'] = 0
    self.logger.info(loss_out)