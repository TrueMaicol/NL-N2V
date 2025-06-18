import os, h5py, glob, argparse, shutil
import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.io import imsave

def train(noisy_path, new_folder_path, image_num):
  """
  Function called by multiple threads. 
  It splits an image in sub-images of default size (512, 512, 3) with 256 pixels overlapped.
    Args:
      - noisy_path (str): path to the noisy images of the dataset
      - new_folder_path (str): path to the new folder where the dataset must be saved
      - image_num (int): index of the image to split
  """
  global crop_size, step
  
  # Retrieve current image and extension
  image_name, extension = os.path.splitext(os.path.basename(noisy_path[image_num]))

  # Retrieve image
  image = h5py.File(noisy_path[image_num], 'r')
  n_img = np.array(image['InoisySRGB']).T
  h,w,c = n_img.shape

  # Images are saved in matlab in float32 in the range [0,1], bring them to uint8
  n_img = n_img * 255
  n_img = n_img.astype(np.uint8)

  # Prepare for cropping
  h_space = np.arange(0, h - crop_size + 1, step)
  if h - (h_space[-1] + crop_size) > 0:
    h_space = np.append(h_space, h - crop_size)
  w_space = np.arange(0, w - crop_size + 1, step)
  if w - (w_space[-1] + crop_size) > 0:
    w_space = np.append(w_space, w - crop_size)
  
  # Crop the image
  idx = 0
  for x in h_space:
    for y in w_space:
      idx += 1
      crop = n_img[x:x+crop_size, y:y+crop_size, :]

      save_path = os.path.join(new_folder_path, f"{image_name}_s{idx:0>3d}.png")
      imsave(save_path, crop)

def test(noisy_path, new_folder_path, image_num):
  """
  Function called by multiple threads. 
  It creates a folder containing the test set indicated by the authors of DND.
    Args:
      - noisy_path (str): path to the noisy images of the dataset
      - new_folder_path (str): path to the new folder where the dataset must be saved
      - image_num (int): index of the image to split
  """
  # Retrieve current image and extension
  image_name, extension = os.path.splitext(os.path.basename(noisy_path[image_num]))

  # Retrieve informations of the .mat file
  infos = h5py.File(os.path.join(data_dir, 'info.mat'), 'r')
  info = infos['info']
  bounding_box = info['boundingboxes']

  # Retrieve image
  image = h5py.File(noisy_path[image_num], 'r')
  n_img = np.array(image['InoisySRGB']).T
  
  # Images are saved in matlab in float32 in the range [0,1], bring them to uint8
  n_img = n_img * 255
  n_img = n_img.astype(np.uint8)

  ref = bounding_box[0][image_num]
  boxes = np.array(info[ref]).T
  for k in range(20):
    idx = [int(boxes[k,0]-1), int(boxes[k,2]), int(boxes[k,1]-1), int(boxes[k,3])]
    crop = n_img[idx[0]:idx[1], idx[2]:idx[3], :].copy()

    save_path = os.path.join(new_folder_path, f"{image_name}_v{j:0>3d}.png")
    imsave(save_path, crop)

def prep_train():
  """
  Function called to prepare the DND dataset for training the method. 
  It splits the dataset in patches of default size (512, 512, 3) with 256 pixels overlapped.
  """
  args = argparse.ArgumentParser()
  args.add_argument('-c', '--crop_size', default=512, type=int)
  args.add_argument('-s', '--step', default=256, type=int)
  args.add_argument('-df', '--dataset_folder', default='path/to/dnd', type=str)
  args.add_argument('-nf', '--new_folder', default='path/to/new/folder', type=str)
  args = args.parse_args()

  # Set crop_size, step and dataset directory
  global crop_size, step, data_dir
  crop_size = args.crop_size
  step = args.step
  data_dir = args.dataset_folder
  new_folder_path = args.new_folder

  assert os.path.exists(data_dir), "The provided folder doesn't exist!"

  train_folder = os.path.join(data_dir, "images_srgb")

  # Extract from the directory a list of all the paths of the images in the dataset
  path_all_noisy = glob.glob(os.path.join(train_folder, '*.mat'))
  path_all_noisy = sorted(path_all_noisy)
  print(f"Number of noisy picture found in the directory: {len(path_all_noisy)}")

  if os.path.exists(new_folder_path):
    shutil.rmtree(new_folder_path, ignore_errors=True)
  os.makedirs(new_folder_path)

  Parallel(n_jobs=10)(delayed(train)(path_all_noisy, new_folder_path, i) for i in tqdm(range(len(path_all_noisy))))

def prep_test():
  """
  Function used to create the test set provided by the authors of DND.
  """
  args = argparse.ArgumentParser()
  args.add_argument('-c', '--crop_size', default=512, type=int)
  args.add_argument('-s', '--step', default=256, type=int)
  args.add_argument('-df', '--dataset_folder', default='path/to/dnd', type=str)
  args.add_argument('-nf', '--new_folder', default='path/to/new/folder', type=str)
  args = args.parse_args()
  
  # Set crop_size, step and dataset directory
  global crop_size, step, data_dir
  crop_size = args.crop_size
  step = args.step
  data_dir = args.dataset_folder
  new_folder_path = args.new_folder

  assert os.path.exists(data_dir), "The provided folder doesn't exist!"
  
  val_folder = os.path.join(data_dir, "images_srgb")
  
  # Extract from the directory a list of all the paths of the images in the dataset
  path_all_noisy = glob.glob(os.path.join(val_folder, '*.mat'))
  path_all_noisy = sorted(path_all_noisy)
  print(f"Number of noisy picture found in the directory: {len(path_all_noisy)}")

  if os.path.exists(new_folder_path):
    shutil.rmtreee(new_folder_path, ignore_errors=True)
  os.makedirs(new_folder_path)

  Parallel(n_jobs=10)(delayed(test)(path_all_noisy, new_folder_path, i) for i in tqdm(range(len(path_all_noisy))))

if __name__ == '__main__':
  prep_train()
  prep_test()