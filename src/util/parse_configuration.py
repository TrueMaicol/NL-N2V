import yaml, os

class ConfigParser():

  def __init__(self, args, extension:str='yaml'):
    """
    Initializer for the configuration parser. 

      Args:
        args: Arguments received from the user, they should contain the name of the yaml file for configuration
        extension (str): String of the extension of the configuration file (default='yaml')
    """
    # load the model configuration
    cfg_file = os.path.join('config', args.config + '.' + extension)
    assert os.path.exists(cfg_file), f"Couldn't find provided file path for the configuration file.\n Got: {cfg_file}"
    with open(cfg_file, 'r') as cfg:
      if extension == 'yaml':
        self.config = yaml.load(cfg, Loader=yaml.BaseLoader)
      else: 
        raise NotImplementedError("For now, the only implemented config file extension is 'yaml'!")
    
    # Add the user given information to the configuration dictionary
    for arg in args.__dict__:
      self.config[arg] = args.__dict__[arg]
    
    # Convert the values corresponding to None strings in actual None elements
    self.convert_none(self.config)
    
  def __getitem__(self, name:str):
    """
    By overwriting the getitem function, I can access the config dictionary from outside the class.

      Args:
        - name (str): key of the item to return.
      
      Returns:
        - value of the config dictionary stored at the given key.
    """
    assert name in self.config, f"Couldn't find the provided key: {name}!"
    return self.config[name]

  def convert_none(self, d:dict):
    """
    Converts the 'None' strings in the config file into actual None objects.

      Args:
        - d (dict): dictionary which 'None' value strings should be converted to actual None type
    """
    for key in d:
      if d[key] == 'None':
        d[key] = None
      if isinstance(d[key], dict):
        self.convert_none(d[key])


if __name__ == '__main__':
  import argparse

  args = argparse.ArgumentParser()
  args.add_argument('-c', '--config', default=None, type=str) 
  args.add_argument('-d', '--device', default=None, type=str) 
  args.add_argument('-r', '--resume', action='store_true')

  args = args.parse_args()

  args.config = "./config/temp.yaml"
  
  cp = ConfigParser(args)
  print(cp) 