import datetime
import os

from ..util.log_msg import LogMsg

class Logger(LogMsg):
  """
  Class that takes care of the logging for the network. 

    Args:
      - max_epoch (int): max epoch for training
      - tot_data (int): num of total data passed in an epoch
      - log_dir (str): path to the directory where log files must be saved (default=None)
      - lol_lvl (str): level of the string to print (default='note')
      - log_file_option (str): type of write to perform on the log file (default='w')
      - log_file_lvl (str): level of the log file (default='info') 
  """
  def __init__(self, max_epoch:int=0, tot_data:int=0, log_dir:str=None, log_lvl:str='note', log_file_option:str='w', log_file_lvl:str='info'):

    self.lvl_list = ['debug', 'note', 'info', 'highlight', 'val']
    self.lvl_color = [bcolors.FAIL, None, None, bcolors.WARNING, bcolors.OKGREEN]

    assert log_lvl in self.lvl_list, "Log level unknown"
    assert log_file_lvl in self.lvl_list, "Log level unknown"
    assert log_file_option == 'w' or log_file_option == 'a', "Log file option not supported"

    # Init log message class
    LogMsg.__init__(self, max_epoch, tot_data)

    # log settings
    self.log_dir = log_dir
    self.log_lvl = self.lvl_list.index(log_lvl)
    self.log_file_lvl = self.lvl_list.index(log_file_lvl)
    
    if self.log_dir is not None:
      # In case the training is resumed, keep the original files
      log_files = os.listdir(log_dir)
      # print(log_dir, len(log_files))
      if len(log_files) != 0:
        # Be carefull: since the log files have a .log extension, cheking only the string 'log' could lead to errors as both of them contains it
        log_file_name = log_files[0] if 'log_' in log_files[0] else log_files[-1]
        self.log_file = open(os.path.join(log_dir, log_file_name), log_file_option)
        val_file_name = log_files[-1] if 'val_' in log_files[-1] else log_files[0]
        self.val_file = open(os.path.join(log_dir, val_file_name), log_file_option)
      else:
        log_time = datetime.datetime.now().strftime('%d-%m-%H-%M')
        self.log_file = open(os.path.join(log_dir, f"log_{log_time}.log"), log_file_option)
        self.val_file = open(os.path.join(log_dir, f"val_{log_time}.log"), log_file_option)
  
  def show(self, msg, lvl_name, end):
    """
    Function that given a message, its level and how it ends, prints it on screen and/or in the log file.
      Args:
        - msg: Message to show
        - lvl_name: Level of the message (used to account for color)
        - end: How the msg should end
    """
    msg = str(msg)
    if self.log_lvl <= lvl_name:
      if self.lvl_color[lvl_name] is not None:
        print("\033[K" + self.lvl_color[lvl_name] + msg + bcolors.ENDC, end=end)
      else:
        print("\033[K" + msg, end=end)
    if self.log_file_lvl <= lvl_name:
      self.write_log_file(msg)
  
  def write_log_file(self, msg:str):
    """
    Function that writes the input message to the log file. 
      Args:
        - msg (str): Message to write in the log_file
    """
    if self.log_dir is not None:
      time = datetime.datetime.now().strftime("%H:%M:%S")
      msg = time + msg
      self.log_file.write(msg + '\n')
      self.log_file.flush()

  # The following functions are all used to set the correct level of the message to show 
  def debug(self, msg, end=None):
    self.show(msg, self.lvl_list.index('debug'), end)

  def note(self, msg, end=None):
    self.show(msg, self.lvl_list.index('note'), end)

  def info(self, msg, end=None):
    self.show(msg, self.lvl_list.index('info'), end)

  def highlight(self, msg, end=None):
    self.show(msg, self.lvl_list.index('highlight'), end)

  def val(self, msg, end=None):
    self.show(msg, self.lvl_list.index('val'), end)
    if self.log_dir is not None:
      self.val_file.write(msg + "\n")
      self.val_file.flush()

  def clear_screen(self):
    if os.name == "nt":
      os.system("cls")
    else:
      os.system("clear")

class bcolors: 
  """
  This class is used to define ANSI colors to print messages with different colors on the CLI
  """
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m' 
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'