import scipy.io as sio
import os, glob, shutil
import numpy as np
from skimage.io import imread

def bundle_submission(sub_folder):
  """
  Function recreated from the code given by the authors of DND to prepare the images for the website's submission.
  """
  out_folder = os.path.join(sub_folder, "bundled/")
  try:
    os.makedirs(out_folder)
  except: pass
  israw = False
  eval_version = "1.0"

  for i in range(50):

    Idenoised = np.zeros((20,), dtype=object)
    for bb in range(20):
      filename = f"{(i+1):04d}_{(bb+1):02d}.mat"
      s = sio.loadmat(os.path.join(sub_folder, filename))
      Idenoised_crop = s['Idenoised_crop']
      Idenoised[bb] = Idenoised_crop
    
    filename = f"{(i+1):04d}.mat"
    sio.savemat(os.path.join(out_folder, filename), {
      "Idenoised": Idenoised,
      "israw": israw,
      "eval_version": eval_version
    })

if __name__ == '__main__':

  # Insert path to the folder with the denoised results
  denoised_folder = "path/to/folder/with/denoised/output"
  assert os.path.isdir(denoised_folder), "The output folder provided doesn't exist."

  test_folder = "path/to/folder/where/you/want/to/put/images/to/submit"
  if os.path.exists(test_folder):
    shutil.rmtree(test_folder, ignore_errors=True)
  os.makedirs(test_folder)

  denoised_paths = glob.glob(os.path.join(denoised_folder, '*DN.png'))
  denoised_paths.sort()

  for i in range(len(denoised_paths)):
    # Get  denoised image
    den_img = imread(denoised_paths[i], as_gray=False)
    # Bring back to [0,1] range
    den_img = np.float32(den_img)
    den_img /= 255

    save_file = os.path.join(test_folder, f"{(i//20)+1:04d}_{(i%20)+1:02d}.mat")
    sio.savemat(save_file, {'Idenoised_crop': den_img})

  bundle_submission(test_folder)