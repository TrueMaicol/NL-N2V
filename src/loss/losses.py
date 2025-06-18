import torch.nn as nn
import torch

from ..loss.self_loss import self_L1, self_L2

class Loss(nn.Module):
  def __init__(self, loss_str:str, tmp_info=[]):
    super().__init__()

    # Set possible loss names
    loss_name_list = ['self_L1', 'self_L2']

    # Remove eventual spaces
    loss_str = loss_str.replace(' ', '')

    # parse loss string
    self.loss_list = []
    for single_loss in loss_str.split('+'):
      weight, name = single_loss.split('*')
      ratio = True if 'r' in weight else False
      # remove eventual ratio flag and retrieve weight
      weight = float(weight.replace('r', ''))

      if name in loss_name_list:
        self.loss_list.append({
            'name' : name,
            'weight': float(weight),
            'func': eval(name)(),
            'ratio': ratio 
          })
      else: 
        raise RuntimeError(f"The provided loss name is unknown. Should be either {loss_name_list[0]} or {loss_name_list[1]}. Got {name}.")
    
    # parse temporal information
    self.tmp_info_list = []
    for name in tmp_info:
      if name in loss_name_list:
        self.tmp_info_list.append({
          'name': name,
          'func': eval(name)()
        })
      else: 
        raise RuntimeError(f"The provided loss name is unknown. Should be either {loss_name_list[0]} or {loss_name_list[1]}. Got {name}.")
    
  def forward(self, input_data, model_output, data, model, ratio=1.0, masked_only:bool=True):
    """
    Forward all the losses and return as dictionary format
      Args:
          input_data: input of the network
          model_output: output of the network
          data: entire data batch
          model: network to be used 
          ratio: (optional) percentage of learning procedure for increasing weight during training
          masked_only (bool): flag for sending masked pixels only to the loss computation or the entire images (default=True)
    """
    if masked_only:
      assert 'mask' in data, "Couldn't find the mask in the data dictionary!"
      # retrieve only the original value of the blind spots. These will be our targets
      input_data = input_data * data['mask']
      # retrieve only the predictions for the values of the blind spots.
      model_output = model_output * data['mask']
    loss_args = (input_data, model_output, data, model)

    # calculate all training losses at one time
    losses = {}
    for single_loss in self.loss_list:
      losses[single_loss['name']] = single_loss['weight'] * single_loss['func'](*loss_args)
      if single_loss['ratio']:
        losses[single_loss['name']] *= ratio
    
    # calculate temporal information
    tmp_info = {}
    for single_tmp_info in self.tmp_info_list:
      with torch.no_grad():
        tmp_info[single_tmp_info['name']] = single_tmp_info['func'](*loss_args)
    
    return losses, tmp_info
    
    

