"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os

"""\
Create a new directory if it doesn't exist -
"""
def createDir(dir):
  try:
    os.mkdir(dir)
    print("Directory '{}' created!".format(dir))
  except:
    print("The directory '{}' already exists!".format(dir))