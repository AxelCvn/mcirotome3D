# -*- coding: utf-8 -*-
import os, sys
import numpy
from PIL import Image
import shutil
from math import *
import glob
import csv
import PIL.ImageOps


directory = "/home/axel/copy_data/"

print" Working in : " + directory + " directory"
#Create new directory to store the new png images
pre, end = os.path.split(directory)
end = end + '_tif'
new_dir = os.path.join(directory,end)
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

for fileName in os.listdir(directory) :
    imgPath = os.path.join(directory,fileName)
    if not os.path.isdir(imgPath):
        os.rename(imgPath, imgPath + '.tif')
