import argparse, os
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.io import imread, imsave
import shutil

crop_size, step = 0

def pipeline(noisy_path, new_folder_path, image_num):
  global crop_size, step
  # Retrieve image path and type of extension
  img_path, extension = os.path.splitext(os.path.basename(noisy_path[image_num]))
  # Retrieve image
  full_img = imread(noisy_path[image_num])
  h,w,c = full_img.shape

  # Prepare for cropping
  h_space = np.arange(0, h - crop_size + 1, step)
  if h - (h_space[-1] + crop_size) > 0:
    h_space = np.append(h_space, h - crop_size)
  w_space = np.arange(0, w - crop_size + 1, step)
  if w - (w_space[-1] + crop_size) > 0:
    w_space = np.append(w_space, w - crop_size)
  
  # Crop the images
  idx = 0
  for x in h_space:
    for y in w_space:
      idx += 1
      # Crop
      cropped_img = full_img[x:x+crop_size, y:y+crop_size, :]
      # Store
      save_path = os.path.join(new_folder_path, "{}_s{:0>3d}{}".format(img_path, idx, extension.lower()))
      imsave(save_path, cropped_img)


def prep():
  args = argparse.ArgumentParser()
  args.add_argument('-c', '--crop_size', default=512, type=int)
  args.add_argument('-s', '--step', default=256, type=int)
  args = args.parse_args()

  # path to the directory that contains the images of the SIDD dataset
  data_dir = "path/to/dataset"

  if not os.path.exists(data_dir):
    raise RuntimeError("The provided path doesn't exist")

  # Extract from the directory a list of all the paths of the noisy images in the dataset
  path_all_noisy = glob(os.path.join(data_dir, "**/NOISY*.PNG"), recursive=True)
  path_all_noisy = sorted(path_all_noisy)
  print(f"Number of noisy picture found in the directory: {len(path_all_noisy)}")

  # In case the folder doesn't exist, build a new one that contains the cropped images
  new_folder_path = "/path/to/new/folder"
  if os.path.exists(new_folder_path):
    shutil.rmtree(new_folder_path, ignore_errors=True)
  os.makedirs(new_folder_path)

  # Set crop_size and step
  global crop_size, step 
  crop_size = args.crop_size
  step = args.step

  Parallel(n_jobs=10)(delayed(pipeline)(path_all_noisy, new_folder_path, i) for i in tqdm(range(len(path_all_noisy))))


if __name__ == "__main__":
  prep()