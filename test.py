import argparse, os

from src.util.parse_configuration import ConfigParser
from src.trainer.new_trainer import StdTrainer

def test(arg:list=None):
  
  # Parse the input configuration
  args = argparse.ArgumentParser()
  args.add_argument('-c', '--config', default=None, type=str)
  args.add_argument('-s', '--run_name', default=None, type=str)
  args.add_argument('-e', '--ckpt_epoch', default=0, type=int)
  args.add_argument('-g', '--gpu', default=None, type=str)
  args.add_argument('--thread', default=4, type=int)
  args.add_argument('--pretrained', default=None, type=str)
  args.add_argument('--self_en', action='store_true')
  args.add_argument('--test_img', default=None, type=str)
  args.add_argument('--test_dir', default=None, type=str)

  # If the arguments are passed through the use of an input list, parse those. Otherwise parse the user provided ones
  if args is not None:
    args = args.parse_args(arg)
  else:
    args = args.parse_args(arg)
    print(args)

  assert args.config is not None, "Configuration is needed!"
  if args.run_name is None:
    # Set the run name to the configuration file's name
    args.run_name = args.config
  
  cfg_dict = ConfigParser(args)

  # Set the device
  if cfg_dict['gpu'] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg_dict['gpu']
  
  # Set the trainer class
  trainer = StdTrainer(cfg_dict)

  # Test over the provided data
  trainer.test()

if __name__ == "__main__":
  arg_list = ['--config=kar0', '--run_name=kar_tr34', '--gpu=0', '--pretrained=kar_tr34_010_psnr_29.44.pth']
  test(arg=arg_list)