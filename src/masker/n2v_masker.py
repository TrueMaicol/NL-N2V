import torch
import torch.nn.functional as F
from ..masker.masker import Masker

class N2V_Masker(Masker):

  def __init__(self, masker_dict: dict):
    """
    Constructor of the masker class. 
    It extends the base class, implementing the functionalities of the original N2V masker.
      Args:
        - masker_dict (dict): dictionary containing the following keys and relative values:
          - strat_name (str): name of the masking strategy to apply
          - pixel_perc (str): percentage of pixels to mask
          - prox_radius (int): radius of the neighborhood of patch used in the masking strategy
    """
    # Check for dictionary keys
    mandatory_keys = ['strat_name', 'pixel_perc', 'prox_radius']
    for key in mandatory_keys:
      assert key in masker_dict, f"Couldn't find key: {key} in the masker dictionary!"

    assert 'strat_name' in masker_dict, "Masking strategy not provided!\n Available strategies: \n- normal_withoutCP; \n- prox_mean; \n- prox_median; \n- uniform_withCP; \n- normal_additive." 

    # set masking strategy and variables
    if masker_dict['strat_name'] == 'normal_withoutCP':
      self.mask_strat = self.normal_withoutCP
    elif masker_dict['strat_name'] == 'prox_mean':
      self.mask_strat = self.prox_mean
    elif masker_dict['strat_name'] == 'prox_median':
      self.mask_strat = self.prox_median
    elif masker_dict['strat_name'] == 'uniform_withCP':
      self.mask_strat = self.uniform_withCP
    elif masker_dict['strat_name'] == 'uniform_withoutCP':
      self.mask_strat = self.uniform_withoutCP
    elif masker_dict['strat_name'] == 'normal_additive':
      self.mask_strat = self.normal_additive
    elif masker_dict['strat_name'] == 'normal_fitted':
      self.mask_strat = self.normal_fitted
    else:
      raise ValueError(f"Weird masking strategy: {masker_dict['strat_name']}.\n Available strategies: \n- normal_withoutCP; \n- prox_mean; \n- prox_median; \n- uniform_withCP; \n- normal_additive.")
    
    self.pixel_perc = float(masker_dict['pixel_perc'])
    self.prox_radius = int(masker_dict['prox_radius']) 

    if 'structN2V' in masker_dict:
      assert 'direction' in masker_dict['structN2V'] and 'edge' in masker_dict['structN2V'], "Missing required parameters!"
      self.direction = masker_dict['structN2V']['direction']
      self.mask_edge = int(masker_dict['structN2V']['edge'])
      self.structn2v = True
    else:
      self.structn2v = False
  
  def mask(self, batch:torch.Tensor) -> torch.Tensor:
    """
    Overide of the mask function. It implements Noise2Void's masking strategy.
      Args:
        - batch (torch.Tensor): batch of images to mask shaped (b,c,h,w)
      
      Returns:
        - batch_res (torch.Tensor): batch of masked images shaped (b,c,h,w)
        - batch_mask (torch.Tensor): batch containing the mask applied to images (b,1,h,w)
    """
    # Extract batch dimensions
    b,c,h,w = batch.shape
    # Set device
    dev = batch.device

    # Init return batches
    batch_res = torch.zeros((b,c,h,w), device=dev)
    batch_mask = torch.zeros((b,1,h,w), device=dev)

    # Temp
    pad_size = self.prox_radius

    assert h == w, f"Image dimension isn't squared! Got: {(h, w)}"
    num_coords = int((h**2) * self.pixel_perc)

    for i in range(b):
      curr_img = batch[i]

      c_img = F.pad(curr_img, (pad_size, pad_size, pad_size, pad_size), "reflect")
      # Resize the h and w coordinates accordingly
      h = c_img.shape[-2]
      w = c_img.shape[-1]
      to_mask = self.rand_unique_coords(self.prox_radius, h - self.prox_radius, num_coords).to(device=dev)
      # print(f"# DEBUG: To mask -> {to_mask}")

      # NOTE: Since different manipulators act differently, it is easier to return the masked_img and mask instead of the coordinates
      masked_img = torch.clone(c_img)
      mask = torch.zeros((1,h,w), device=dev)

      pixels = self.mask_strat(c_img, to_mask)
      # DEBUGG: print(pixels)

      # Substitute the single points in the masked image and generate the mask! 
      masked_img[:, to_mask[:, 0], to_mask[:, 1]] = pixels  #c_img[:, new_coords[:, 0], new_coords[:, 1]]
      mask[:, to_mask[:, 0], to_mask[:, 1]] = 1

      if self.structn2v:
        # Mask the pixels around the central ones
        self.structn2v_mask(masked_img, to_mask)

      batch_res[i] = masked_img[:, pad_size:-pad_size, pad_size:-pad_size]
      batch_mask[i] = mask[:, pad_size:-pad_size, pad_size:-pad_size]
    return batch_res, batch_mask

  def normal_withoutCP(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Normal_withoutCP manipulator: generates the coordinate of the masking pixel by applying a normal distribution

      Args:
        - img (torch.Tensor): image to mask
        - to_mask (torch.Tensor): coordinates of the pixels to mask

      Returns:
        - tensor containing the values of the pixels to substitute
    """
    c,h,w = img.shape
    # Generate the coordinates of the points to substitute
    cnt = 0
    new_coords = torch.empty((0,2), dtype=torch.int32)

    while new_coords.shape[0] != to_mask.shape[0]:
      # generate random point, make sure it doesn't escape the image's limits
      coord = torch.round(torch.normal(mean=to_mask[cnt,:].float(), std=float(self.prox_radius)))
      coord = torch.clamp(coord, 0, h - 1).to(device=new_coords.device)
      coord = coord[None,:]

      if all(torch.linalg.norm(new_coords - coord, dim=1) != 0):
        new_coords = torch.cat((new_coords, coord.type(torch.uint8)), dim=0)
        cnt += 1

    return img[:, new_coords[:, 0], new_coords[:, 1]]

  def prox_mean(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that computes the mean values of the neighborhood of the pixels to mask.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - tensor containing the values of the pixels to substitute
    """
    ptcs = self.gather_patches(img, to_mask)
    ptcs_wo_ctr = self.exclude_ctr(ptcs)
    
    values = torch.mean(ptcs_wo_ctr, dim=-1)
    return values

  def prox_median(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that computes the median values of the neighborhood of the pixels to mask.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - tensor containing the values of the pixels to substitute
    """
    ptcs = self.gather_patches(img, to_mask)
    ptcs_wo_ctr = self.exclude_ctr(ptcs)

    values = torch.median(ptcs_wo_ctr, dim=-1).values
    return values
  
  def uniform_withCP(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that substitute the pixels to mask with a random one it its proximity.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - tensor containing the values of the pixels to substitute
    """
    ptcs = self.gather_patches(img, to_mask)
    rand_pixels = self.gather_rand_pixels(ptcs.view(ptcs.size(0), ptcs.size(1), -1))

    return rand_pixels
  
  def uniform_withoutCP(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that substitute the pixels to mask with a random one it its proximity. It doesn't account for the pixels to mask themselves.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - tensor containing the values of the pixels to substitute
    """

    ptcs = self.gather_patches(img, to_mask)
    ptcs_wo_ctr = self.exclude_ctr(ptcs)
    rand_pixels = self.gather_rand_pixels(ptcs_wo_ctr.view(ptcs_wo_ctr.size(0), ptcs_wo_ctr.size(1), -1))

    return rand_pixels

  def normal_additive(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that generates the values of the pixels to mask by adding or subtracting to the original values some values obtained from a normal distribution.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - tensor containing the values of the pixels to substitute
    """
    img_pxls = img[:, to_mask[:, 0], to_mask[:, 1]]
    pixels = torch.normal(mean=img_pxls, std=float(self.prox_radius))
    return pixels
  
  def normal_fitted(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that generates the values of the pixels to mask utilizing a normal function with as mean the mean of the pixels proximity and as std the std of the pixels proximity.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - tensor containing the values of the pixels to substitute
    """
    # TODO: This function returns negative values sometimes! Check why and ask Edo!
    ptcs = self.gather_patches(img, to_mask)
    pixels = torch.normal(mean=torch.mean(ptcs, dim=(-2, -1)), std=torch.std(ptcs, dim=(-2, -1)))
    # print(f"Std: {torch.std(ptcs, dim=(-2,-1))}")
    return pixels

  def gather_patches(self, img:torch.Tensor, to_mask:torch.Tensor) -> torch.Tensor:
    """
    Function that given an image and some pixels to mask, returns a batch of patches around those pixels.

      Args:
        - img (torch.Tensor): Image to mask
        - to_mask (torch.Tensor): Coordinates of the pixels to mask
      
      Returns:
        - ptcs (torch.Tensor): Batch of sub-patches around the pixels in to_mask
    """
    c,h,w = img.size() 
    top_left = torch.clamp(to_mask - self.prox_radius, 0, h)
    bottom_right = torch.clamp(to_mask + self.prox_radius + 1, 0, w)
    ptcs = []

    # Cut the references from the image
    for i in range(to_mask.shape[0]):
      patch = img[:, top_left[i, 0]:bottom_right[i, 0], top_left[i, 1]:bottom_right[i, 1]]
      ptcs.append(patch)
    
    # Convert to tensor
    ptcs = torch.stack(ptcs, dim=1)
    return ptcs

  def exclude_ctr(self, batch:torch.Tensor) -> torch.Tensor:
    """
    Function that given a batch of images, returns the batch without accounting for the images' central pixels.
      Args:
        - batch (torch.Tensor): Batch of images of dimension (n,c,h,w)
      
      Returns:
        - batch_wo_ctr (torch.Tensor): Batch without the central pixels, with the last 2 dimensions flattened (n,c,h*w) 
    """
    n,c,h,w = batch.shape

    # Create a mask to exclude the center
    mask = torch.ones_like(batch, dtype=torch.bool)
    mask[:,:, h//2, w//2] = False

    return batch[mask].view(n, c, -1)
  
  def gather_rand_pixels(self, batch:torch.Tensor) -> torch.Tensor:
    """
    Function that given a batch of flattened batches, returns a ransom pixel inside the patch.

      Args:
        - batch (torch.Tensor): batch of flattened images (c, n, h*w)
      
      Returns:
        - pixels (torch.Tensor): random pixels values for each row (n) 
    """
    c,n,hw = batch.size()

    idxs = torch.randint(0, hw, (n,))
    btc_ptchs = torch.arange(n)

    pixels = batch[:, btc_ptchs, idxs]
    return pixels

  def structn2v_mask(self, img:torch.Tensor, coords:torch.Tensor):
    """
    Function that masks the image with the same effectiveness of StructN2V.

      Args:
        - img (torch.Tensor): image to mask
        - coords (torch.Tensor): coordinates of the masked pixels
    """
    # Create mask
    mask = torch.ones(self.mask_edge*2+1, dtype=torch.bool)
    if self.direction == 'V':
      mask = mask[:, None]
    elif self.direction == 'H':
      mask = mask[None, :]
    elif self.direction == 'Q':
      mask = torch.ones((self.mask_edge*2+1, self.mask_edge*2+1), dtype=torch.bool)
    else: 
      raise RuntimeError("Unexpected mask direction provided!")
    center = torch.tensor(mask.size()) // 2
    mask[tuple(center.tolist())] = False

    ndim = mask.ndim
    
    # Compute displacements from the center
    idxs = torch.stack(torch.meshgrid([torch.arange(s) for s in mask.size()], indexing='ij'))
    dx = idxs[:, mask] - center[:, None]
    dx = dx.to(device=coords.device)

    # Combine all coords with displacements
    mix = (dx.T[..., None] + coords.T[None]).permute(1, 0, 2).reshape(ndim, -1).T

    # Stay within the image boundary
    max_vals = torch.tensor(img.shape[-1], device=coords.device) - 1
    mix = torch.clamp(mix, min=0, max=max_vals).long()

    # Replace neighboring pixels in all channels with random values
    img[:, mix[:, 0], mix[:, 1]] = torch.rand((img.shape[0], mix.shape[0]), device=img.device)


if __name__ == '__main__':
  # Test the rewritten manipulators
  imsz = (3, 30, 30)
  low = 0
  high = 256
  fake_img = torch.randint(low=low, high=high, size=imsz)
  fake_img = fake_img.type(torch.float32)

  print(fake_img)
  # # print(fake_img.shape)
  fake_img = fake_img[None, :, :, :]
  # # print(fake_img.shape)

  masker_dict = {
    # 'strat_name': 'prox_mean',
    # 'strat_name': 'normal_withoutCP',
    # 'strat_name': 'prox_median',
    # 'strat_name': 'uniform_withCP',
    # 'strat_name': 'uniform_withoutCP',
    # 'strat_name': 'normal_additive',
    'strat_name': 'normal_fitted',
    'pixel_perc': 0.0023,
    'prox_radius': 3,
    'structN2V': {
      'direction': 'Q',
      'edge': 3
    }
  }
  my_masker = N2V_Masker(masker_dict)
  res, mask = my_masker.mask(fake_img)

  print(res.shape, mask.shape)
  print(res)
  # print(mask)
