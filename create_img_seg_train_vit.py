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

print(len(images))

image_set_name = "test"
logdir = "Logs"
method_classes = [neural_method]
methods=[m() for m in method_classes]
create_databases(images,methods,image_set_name)
aberrations = aberrations_2
run_in_chunks9(methods, images, aberrations, chunk_size=100, weights=[],
    logdir=logdir)

# modify aberrations and run_in_chunks 11 to augment data
print("Done")

