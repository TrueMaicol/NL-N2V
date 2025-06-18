import torch
from ..masker.masker import Masker

class FastMasker(Masker):
  """
  Class that overrides the mask() function of the original masker to make it run faster over the input data.
  It is meant for training using a Downsampling operation, this is an initial attempt that has not been tested much yet.
  """
  def __init__(self, masker_dict:dict):
    super().__init__(masker_dict)
  
  def mask(self, batch:torch.Tensor, sf:int) -> torch.Tensor:
    """
    Override of the original masker's function.
    It keeps the inputs and outputs, but performs the self-similarity search ony one time for each image in the original batch of n images.
      Args:
        - batch (torch.Tensor): Batch of images to mask shaped (b, c, h, w)
        - sf (int): stride factor applied to the images
      
      Returns:
        - batch_res (torch.Tensor): Batch of masked images shaped (b, c, h, w)
        - batch_mask (torch.Tensor): Batch containing the mask applied to images (b, 1, h, w)
    """
    b,c,h,w = batch.shape
    # Set device that will hold the computation
    dev = batch.device
    hspe = self.sp_edge // 2
    # Init the batch that will contain the returned masked images
    batch_res = torch.zeros((b,c,h,w), device=dev)
    # Init the batch that will contain the mask applied to the images
    batch_mask = torch.zeros((b,1,h,w), device=dev)
    # Compute the number of pixels to mask given the input image's size 
    assert h == w, f"Image dimension isn't squared! Got: {(h, w)}"
    num_coords = int((h**2) * self.pixel_perc)

    # Iterate through all the images in the batch
    for i in range(b):
      # Extract the current image
      curr_img = batch[i]
      # Clone the current image for computations
      cloned_curr = torch.clone(curr_img)

      # Create a copy of the image to mask
      masked_img = torch.clone(curr_img)
      # Create a mask of zeros with the same size of the current image in consideration
      mask = torch.zeros((1,h,w), device=dev)
      
      # Once every 25 images
      if i % (sf**2) != 0:
        # Re-use the same pixels to mask and prediction of the first image of the batch    
        assert len(to_mask) == len(sim_coords), f"The points to mask and the similar_coordinates drawn are not in the same number.\n Got, points to mask: {len(to_mask)}, sim_coords: {len(sim_coords)}"
      else:
        # Randomly generate the points to mask for each subpatch
        to_mask = self.rand_unique_coords(hspe, h - hspe, num_coords)

        # Build the windows where self-similarity needs to be searched
        windows = cloned_curr.unsqueeze(0).expand(to_mask.shape[0], c, h, w)

        # Compute the top_left and bottom_right points with respect to the reference points to mask
        ref_top_left = to_mask - hspe
        ref_bottom_right = to_mask + hspe + 1
        ref_sps = []

        # Cut the reference patches and put them in a list
        for j in range(to_mask.shape[0]):
          ref_patch = cloned_curr[:, ref_top_left[j, 0]:ref_bottom_right[j, 0], ref_top_left[j, 1]:ref_bottom_right[j, 1]]
          ref_sps.append(ref_patch)
        
        # Convert the references to a tensor
        ref_sps = torch.stack(ref_sps)

        # Compute the self-similar coordinates with respect to the references 
        sim_coords = self.find_sim_coords(windows, ref_sps)

      # Compute the displacements from the central pixel
      patch_idxs = torch.arange(-hspe, hspe + 1)

      # Compute the grids over which the displacements are calculated
      grid_h, grid_w = torch.meshgrid(patch_idxs, patch_idxs, indexing='ij')
      grid_h = grid_h.to(device=dev)
      grid_w = grid_w.to(device=dev)

      if not self.ctr_only:
        # Find the indices of the pixels to substitute, in matrix form
        patch_h = grid_h.unsqueeze(-1) + to_mask[:, 0]
        patch_w = grid_w.unsqueeze(-1) + to_mask[:, 1]

        # Find the indices of the pixels to swap over the ones to substitute, in matrix form
        sim_patch_h = grid_h.unsqueeze(-1) + sim_coords[:, 0]
        sim_patch_w = grid_w.unsqueeze(-1) + sim_coords[:, 1]

        # Use advanced indexing to perform masking
        masked_img[:, patch_h, patch_w] = curr_img[:, sim_patch_h, sim_patch_w]
        mask[:, patch_h, patch_w] = 1
      else: 
        masked_img[:, to_mask[:, 0], to_mask[:, 1]] = curr_img[:, sim_coords[:, 0], sim_coords[:, 1]]
        mask[:, to_mask[:, 0], to_mask[:, 1]] = 1

      batch_res[i] = masked_img
      batch_mask[i] = mask
    return batch_res, batch_mask 