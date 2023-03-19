import sys
import os
import h5py
import pickle
import math
import numpy as np
import pandas as pd
import collections
import random
import mahotas
from sklearn.preprocessing import normalize
from PIL import Image
import imutils
import logging

hdf5_file = h5py.File('data/LLD-icon.hdf5', 'r')
images = hdf5_file['data']
images = images[int(len(images) * 0.2):int(len(images) * 0.21)]
images = [np.transpose(i) if i.shape[0] == 3 else i for i in images]

count = 0

for i in range(len(images)):
    img = Image.fromarray(images[i])
    count += 1
    img.save('./gan_data/img/' + str(count) + '.jpg')
