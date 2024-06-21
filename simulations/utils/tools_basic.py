"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os


def createDir(dir):
  """
  Create a new directory if it doesn't exist -
  """
  try:
    os.mkdir(dir)
    print("Directory '{}' created!".format(dir))
  except:
    print("The directory '{}' already exists!".format(dir))