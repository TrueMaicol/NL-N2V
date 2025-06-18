import os
from importlib import import_module

for module in os.listdir(os.path.dirname(__file__)):
  # skip files that are the init or are not a python file
  if module == '__init__.py' or module[-3:] == '.py' or module == '__pycache__':
    continue
  import_module(f'src.masker.{module[:-3]}')

del module