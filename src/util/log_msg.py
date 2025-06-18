import time, datetime

class LogMsg():
  """
    Class that handles the messages the logger will be storing during the network training.
  """

  def __init__ (self, max_epoch:int, tot_data:int, min_time_interval:float=0.2):
    """
    To init the class, two parameters are necessary.
      Args:
          max_epoch (int): Max epoch for training
          tot_data (int): Num of total data passed in an epoch
          min_time_interval (float): Min number of seconds between a message and the next one (default = 0.2)
    """
    # Set the starting time of the message
    self.start_time = time.time()
    self.log_time = self.start_time

    self.max_epoch = max_epoch
    self.tot_data = tot_data
    self.min_time_interval = min_time_interval
  
  def start(self, start_epoch:int, start_data:int):
    """
    Function that starts the time to print messages accounting for the training progress.
      Args: 
        - start_epoch (int): first epoch of the current training
        - start_data (int): image index to read
    """

    self.start_epoch = start_epoch
    self.start_data = start_data
    self.current_epoch = start_epoch
    self.start_time = time.time()
    self.prog_time = self.start_time
  
  def compute_progress(self, current_epoch:int, current_data:int):

    self.prog_time = time.time()

    assert current_epoch <= self.max_epoch, "Current epoch value should be less or equal than max one"
    assert current_data <= self.tot_data, "Current data value should be less or equal than total one"

    start_percentage = 0
    start_percentage = (self.start_data/self.tot_data + self.start_epoch)/self.max_epoch
    start_percentage *= 100

    prog_percentage = 0
    prog_percentage = (current_data/self.tot_data + current_epoch)/self.max_epoch
    prog_percentage *= 100

    prog_percentage = (prog_percentage - start_percentage) / (100 - start_percentage) * 100

    time_passed = time.time() - self.start_time
    time_passed_str = str(datetime.timedelta(seconds=int(time_passed)))
    
    if prog_percentage != 0:
      tot_time = 100 * time_passed/prog_percentage
      time_remaining = tot_time - time_passed
      time_remaining_str = str(datetime.timedelta(seconds=int(time_remaining)))
      time_tot_str = str(datetime.timedelta(seconds=int(tot_time)))
    else:
      time_tot_str = 'INF'
      time_remaining_str = 'INF'
    
    return prog_percentage, time_passed_str, time_remaining_str, time_tot_str

  def print_log_msg(self, current_epoch, current_data):
    if time.time() - self.prog_time >= self.min_time_interval:
      pg_perc, tp_str, tr_str, tt_str = self.compute_progress(current_epoch, current_data)

      msg = f"\033[K>>> progress: {pg_perc:.2f}%, time_passed: {tp_str}, time_remaining: {tr_str}, tot_time: {tt_str} \t\t\t\t\t"

      print("\r", end='')
      print(msg, end='', flush=True)

      return msg.replace('\t', '')
    return
  
  def get_start_msg(self):
    return ' Start >>>'

  def get_end_msg(self):
    tot_time = time.time() - self.start_time
    time_tot_str = str(datetime.timedelta(seconds=int(tot_time)))
    msg = f"End >>> total time  elapsed: {time_tot_str}"
    return msg


def test():
  import logging

  logging.basicConfig(format="%(message)s",
                      level=logging.INFO, 
                      handlers=[logging.StreamHandler()])
  
  lm = LogMsg(10, 10)
  se = 0
  sd = 0

  lm.start(se, sd)

  for i in range(10):
    for j in range(10):
      for k in range(10):
        time.sleep(0.5)
        lm.print_log_msg(i, j)
      logging.info('ttt')

if __name__ == "__main__":
  test()