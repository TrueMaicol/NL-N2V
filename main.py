import argparse, os
from datetime import datetime

from src.util.parse_configuration import ConfigParser
from src.trainer.new_trainer import StdTrainer

def main(arg:list=None):

  # Parse the input configuration
  args = argparse.ArgumentParser()
  args.add_argument('-c', '--config', default=None, type=str)
  args.add_argument('-s', '--run_name', default=None, type=str)
  args.add_argument('-r', '--resume', action='store_true')
  args.add_argument('-g', '--gpu', default=None, type=str)
  args.add_argument('--thread', default=4, type=int)
  if args is not None:
    args = args.parse_args(arg)
  else:
    # args = args.parse_args(arg)
    print("Problem with arguments!")

  # Check the user provided a configuration file 
  assert args.config is not None, 'config file is needed!'

  # If the run name is not provided, assign it
  if args.run_name is None:
    time_stmp = datetime.now().strftime("%d-%m-%Y-%H-%M")
    args.run_name = args.config + '_' + time_stmp 

  cfg_dict = ConfigParser(args)
  # print(cfg_dict['config'], cfg_dict['run_name'])

  # Set the device 
  if cfg_dict['gpu'] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg_dict['gpu']

  # Define a trainer class
  trainer = StdTrainer(cfg_dict)

  # call the trainer to train the network
  trainer.train()

if __name__ == "__main__":
  main()
  