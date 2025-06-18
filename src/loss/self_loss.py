import torch.nn.functional as F

class self_L1():
  def __call__(self, input_data, model_output, data, model):
    output = model_output
    target_noisy = input_data # Could also be input_data
    return F.l1_loss(output, target_noisy)

class self_L2():
  def __call__(self, input_data, model_output, data, model):
    output = model_output
    target_noisy = input_data # Could also be input_data 
    return F.mse_loss(output, target_noisy)
  