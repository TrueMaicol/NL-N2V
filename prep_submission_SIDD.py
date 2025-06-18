import os, scipy.io, base64, glob
import numpy as np
import pandas as pd
from skimage.io import imread

def array_to_base64string(x:np.ndarray):
  array_bytes = x.tobytes()
  base64_bytes = base64.b64encode(array_bytes)
  base64_string = base64_bytes.decode('utf-8')
  return base64_string

def base64string_to_array(base64string, array_dtype, array_shape):
  decoded_bytes = base64.b64decode(base64string)
  decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
  decoded_array = decoded_array.reshape(array_shape)
  return decoded_array

if __name__ == '__main__':

  test_folder = 'path/to/test/images'
  # Open denoised folder
  denoised_folder = 'path/to/denoised/test/images'

  denoised_images = glob.glob(os.path.join(denoised_folder, '*DN.png'))
  denoised_images.sort()
  print(len(denoised_images), denoised_images[0])

  # Find input and open it
  inputs = scipy.io.loadmat(os.path.join(test_folder, 'BenchmarkNoisyBlocksSrgb'))
  inputs = inputs['BenchmarkNoisyBlocksSrgb']
  print(inputs.shape)

  # Create output block
  output_blocks_base64strings = []
  for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
      in_img = inputs[i, j, :, :, :]
      out_img = imread(denoised_images[i*32 + j])
      assert in_img.shape == out_img.shape
      assert in_img.dtype == out_img.dtype
      out_img_base64string = array_to_base64string(out_img)
      output_blocks_base64strings.append(out_img_base64string)

  # Save outputs on a .csv file
  output_file = 'output_name.csv'
  print(f'Saving outputs to {output_file}')
  output_df = pd.DataFrame()
  n_blocks = len(output_blocks_base64strings)
  print(f'Number of blocks: {n_blocks}')
  output_df['ID'] = np.arange(n_blocks)
  output_df['BLOCK'] = output_blocks_base64strings

  output_df.to_csv(output_file, index=False)

