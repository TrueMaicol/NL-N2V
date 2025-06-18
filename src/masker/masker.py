import torch
import torch.nn.functional as F

class Masker():
  def __init__(self, masker_dict:dict):
    """
    Constructor of the maker class
      Args:
        - masker_dict (dict): dictionary containing the following keys and relative values:
          - sim_patch_edge (int): dimension of the edge of the patch to use for self-similarity computation
          - repl_patch_edge (int): dimension of the edge of the patch to mask
          - speedup_factor (int): number of pixels to skip when searching for self-similarity
          - pixel_perc (float): percentage of pixels to mask in each image
          - ctr_only (bool): decider for the masking of a single pixel or the entire sub-patch (default=False)
    """
    mandatory_keys = ['sim_patch_edge', 'repl_patch_edge', 'speedup_factor', 'pixel_perc', 'center_only']
    for key in mandatory_keys:
      assert key in masker_dict, f"Couldn't find key: {key} in the masker dictionary!"
    
    self.spe = int(masker_dict['sim_patch_edge'])
    self.rpe = int(masker_dict['repl_patch_edge'])
    self.spf = int(masker_dict['speedup_factor'])
    self.pixel_perc = float(masker_dict['pixel_perc'])
    
    if masker_dict['center_only'] == 'False':
      self.ctr_only = False
    elif masker_dict['center_only'] == 'True':
      self.ctr_only = True
    else:
      raise ValueError(f"Unusual value for center_only flag for the masker.\n Got: {masker_dict['center_only']}")
    
    # Depending on the value of the similarity patch, decide wether to pad or not
    if self.spe > self.rpe:
      self.to_pad = True
    elif self.spe == self.rpe:
      self.to_pad = False
    else:
      raise ValueError(f"Unexpected value of sim_patch_edge, which is lower than that of repl_patch_edge!")
  
  def rand_unique_coords(self, min:int, max:int, num:int, dev:str='gpu') -> torch.Tensor:
    """Function that generates random num number of indexes of pixels to mask without duplicates. It isn't completely random since it takes into consideration a minimum distance between each point.
    For now the minimum distance is set to 12, and it is calculated with the Pitagora's theorem.

      Args:
        - min (int): min value of an index
        - max (int): max value of an index
        - num (int): number of indexes to generate
        - dev (str): either gpu or cpu, depending on the type of masker to use (default=gpu)

      Returns:
        - coordinates (torch.Tensor): array of coordinates of the pixels to mask
    """
    min_dist = 12
    coordinates = torch.empty((0, 2), dtype=torch.int64)
    
    # iterate until the number of generated points is the desired one
    while coordinates.shape[0] < num:
      # generate random point 
      coord = torch.randint(min, max, (1, 2), dtype=torch.float32)
      # compute distance to the other drawn points and if greater then min append point
      if all(torch.linalg.norm(coordinates - coord, dim=1) >= min_dist):
        coordinates = torch.cat((coordinates, coord.type(torch.int64)), dim=0)
    
    assert coordinates.shape[0] == num, "Weird ending shape of the loop!"

    # Send the result to the correct device
    if dev == 'gpu':
      coordinates = coordinates.cuda()
    elif dev == 'cpu':
      coordinates = coordinates.cpu()
    else:
      raise RuntimeError(f"Unknown device provided. Got: {dev}")
    
    return coordinates
     
  def find_sim_coords(self, window:torch.Tensor, ref_sp:torch.Tensor) -> torch.Tensor:
    """
    Function that given a batch of reference patches and a batch of images, returns the coordinates of the most similar patches with respect to the references in the images. The size of the searched similar patches are the same of the references one. Moreover, the function uses a speedup factor of 2 to speed up the overall computation.
      
      Args:
        - window (torch.Tensor): Parts of the image where to search the self-similarity for each reference (shape: b,c,h,w)
        - ref_sp (torch.Tensor): Reference patches for which the most similar must be found (shape: b,c,h',w')

      Returns:
        - win_coords (torch.Tensor): Coordinates of the centers of the most similar patches found in the windows
    """
    # Extract channels dimension
    c = window.shape[1]
    # Build a volume of all the patches to confront with the reference
    ptc_vol = window.unfold(-2, self.spe, self.spf).unfold(-2, self.spe, self.spf)
    # Move the channels to last dimension
    ptc_vol = ptc_vol.permute(0,2,3,4,5,1).contiguous()

    # Move channels to last dimension
    ref_sp = ref_sp.permute(0,2,3,1)
    # Build a volume containing only the references, with the same size as the patches volume to perform fast difference
    ref_vol = ref_sp.unsqueeze(1).unsqueeze(1).expand_as(ptc_vol)

    assert ptc_vol.shape == ref_vol.shape, f"The patches to confront and the reference do not have the same dimensions. Got patches dim: {ptc_vol.shape}, ref dim: {ref_vol.shape}"

    # Since images are stored in uint8, cast them to float32 before performing the difference to avoid problems
    ptc_vol = ptc_vol.type(torch.float32)
    ref_vol = ref_vol.type(torch.float32)

    # Calculate the difference between the volumes and square it to remove negative values
    sqr_diff = (ptc_vol - ref_vol) ** 2

    # calculate the L2 norm of these squared differences to find the distance between the reference and each patch in the window
    distances = sqr_diff.sum(dim=(-3,-2,-1))**2 / (self.spe**2 * c)

    # Remove trivial solution if present
    distances = torch.where(distances > 0., distances, float('inf'))
    # Flatten the last 2 dimensions and compute the argmin for each reference patch
    idxs = torch.argmin(distances.view(distances.shape[0], -1), dim=1)
    # Convert the indices in 2D coordinates
    h_coords = idxs // distances.shape[1]
    w_coords = idxs % distances.shape[2]

    # Compute the indices of the centers of the most similar sub-patches found with respect to the references
    final_coords = torch.stack((h_coords * self.spf + self.spe // 2, w_coords * self.spf + self.spe // 2), dim=1)
    return final_coords

  def mask(self, batch:torch.Tensor, sf:int=None) -> torch.Tensor:
    """
    Function used to mask a batch of images. This masking is performed by randomly generating a specific number of pixels, sent as input during the Masker initialization, then calculating the most similar subpatch to the one having that pixel as center and substituting it to the original one.
      Args:
        - batch (torch.Tensor): Batch of images to mask shaped (b, c, h, w)
        - sf (int): stride factor applied to the batch (default=None)
      
      Returns:
        - batch_res (torch.Tensor): Batch of masked images shaped (b, c, h, w)
        - batch_mask (torch.Tensor): Batch containing the mask applied to images (b, 1, h, w)
    """
    # Extract batch dimensions
    b,c,h,w = batch.shape
    # Set device that will hold the computation
    dev = batch.device
    # Half similar patch edge
    hspe = self.spe // 2
    # Half replacement patch edge
    hrpe = self.rpe // 2
    # Compute pad_size
    pad_size = hspe - hrpe
    # Init the batch to return containing the masked images
    batch_res = torch.zeros((b,c,h,w), device=dev)
    # Init the batch to return containing the masks
    batch_mask = torch.zeros((b,1,h,w), device=dev)
    # calculate the number of pixels to mask given the input image's size
    assert h == w, f"Image dimension isn't squared! Got: {(h, w)}"
    num_coords = int((h**2) * self.pixel_perc)

    # Iterate through all the images in the batch
    for i in range(b):
      # Extract current image
      c_img = batch[i]
      
      # Pad the input image if needed
      if self.to_pad:
        # TODO: Try different pad styles
        curr_img = F.pad(c_img, (pad_size, pad_size, pad_size, pad_size), "reflect")
        # Resize the h and w coordinates accordingly
        h = curr_img.shape[-2]
        w = curr_img.shape[-1]
      else:
        curr_img = c_img

      # Clone the current image for computations
      cloned_curr = torch.clone(curr_img)
      
      # Randomly generate the points to mask
      to_mask = self.rand_unique_coords(hspe, h - hspe, num_coords)

      # Create a copy of the image to mask
      masked_img = torch.clone(curr_img)
      # Create a mask of zeros with the same size of the current image in consideration
      mask = torch.zeros((1, h, w), device=dev)

      # Build the windows where self-similarity needs to be searched
      # NOTE: The searching window is not yet implemented due to the difficulty in stacking the exact shapes
      windows = cloned_curr.unsqueeze(0).expand(to_mask.shape[0], c, h, w)

      # Compute the top_left and bottom_right points with respect to the reference points to mask
      ref_top_left = to_mask - hspe
      ref_bottom_right = to_mask + hspe + 1
      ref_sps = []

      # Cut the reference patches and put them in a list
      for j in range(to_mask.shape[0]):
        ref_patch = cloned_curr[:, ref_top_left[j, 0]:ref_bottom_right[j, 0], ref_top_left[j, 1]:ref_bottom_right[j, 1]]
        ref_sps.append(ref_patch)
      
      # Convert from list to torch.Tensor
      ref_sps = torch.stack(ref_sps)

      # Compute the most similar coordinates with respect to the references
      sim_coords = self.find_sim_coords(windows, ref_sps).to(to_mask.device)

      # Compute the displacements from the central pixel
      patch_indices = torch.arange(-hrpe, hrpe + 1)

      # Compute the grids over which the displacements are calculated
      grid_h, grid_w = torch.meshgrid(patch_indices, patch_indices, indexing='ij')
      grid_h = grid_h.to(device=dev)
      grid_w = grid_w.to(device=dev)

      if not self.ctr_only:
        # Find the indices of the pixels to substitute, in matrix form
        patch_h = grid_h.unsqueeze(-1) + to_mask[:, 0]
        patch_w = grid_w.unsqueeze(-1) + to_mask[:, 1]

        # Find the indices of the pixels to swap over the ones to substitute, in matrix form
        sim_patch_h = grid_h.unsqueeze(-1) + sim_coords[:, 0]
        sim_patch_w = grid_w.unsqueeze(-1) + sim_coords[:, 1]

        # Replace the pixels to substitute with the chosen ones
        masked_img[:, patch_h, patch_w] = curr_img[:, sim_patch_h, sim_patch_w]
        mask[:, patch_h, patch_w] = 1
      else:
        masked_img[:, to_mask[:, 0], to_mask[:, 1]] = curr_img[:, sim_coords[:, 0], sim_coords[:, 1]]
        mask[:, to_mask[:, 0], to_mask[:, 1]] = 1
      
      # print(f"End masking l: {datetime.datetime.now()}")
      if self.to_pad:
        # Remove applied masking
        batch_res[i] = masked_img[:, pad_size:-pad_size, pad_size:-pad_size]
        batch_mask[i] = mask[:, pad_size:-pad_size, pad_size:-pad_size]
      else:
        batch_res[i] = masked_img
        batch_mask[i] = mask
    return batch_res, batch_mask