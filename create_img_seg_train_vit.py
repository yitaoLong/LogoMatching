# combined methods for the whole images methods
import sys
import os
import h5py
import pickle
from matplotlib import pyplot as plt

from scipy import ndimage as ndi
from scipy import stats as sstats

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from random import shuffle
from tqdm import tqdm, tnrange, tqdm_notebook
import collections
import random
import mahotas
from sklearn.preprocessing import normalize
from PIL import Image
import imutils
import logging

from glob import glob

import warnings
warnings.filterwarnings("ignore")

from libseg.icon_util import *
from libseg.methods import *
from libseg.aberrations import *

hdf5_file = h5py.File('data/LLD-icon.hdf5', 'r')
images = hdf5_file['data']
images = images[int(len(images) * 0.2):int(len(images) * 0.22)]
images = [np.transpose(i) if i.shape[0] == 3 else i for i in images]

#img_files=glob('data/ut-zap50k-images-square/shoes/**/*.jpg', recursive=True)
#images2 = [cv2.resize(i, (32, 32)) for i in map(cv2.imread, img_files[int(len(img_files) * 0.2):int(len(img_files) * 0.22)]) if i is not None]
## transpose the images because they're stored in a weird color channel first format, as indicated by shape[0] being 3
#images2 = [np.transpose(i) if i.shape[0] == 3 else i for i in images2]
#
#images.extend(images2)
print(len(images))

image_set_name = "test"
logdir = "Logs"


method_classes = [neural_method]
methods=[m() for m in method_classes]
create_databases(images,methods,image_set_name)
run_in_chunks8(methods, images, aberrations, chunk_size=100, weights=[],
    logdir=logdir)
print("Done")

