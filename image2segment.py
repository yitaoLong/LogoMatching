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
images = images[:int(len(images) * 0.004)]
images = [np.transpose(i) if i.shape[0] == 3 else i for i in images]

img_files=glob('data/ut-zap50k-images-square/shoes/**/*.jpg', recursive=True)
images2 = [cv2.resize(i, (32, 32)) for i in map(cv2.imread, img_files[:int(len(img_files) * 0.004)]) if i is not None]
# transpose the images because they're stored in a weird color channel first format, as indicated by shape[0] being 3
images2 = [np.transpose(i) if i.shape[0] == 3 else i for i in images2]

images.extend(images2)
print(len(images))

image_set_name = "segment"

logdir = "image2segment_single_worst"
if not os.path.exists("Logs"):
    os.mkdir("Logs")
if not os.path.exists("Logs/"+logdir):
        os.mkdir("Logs/"+logdir)

method_classes = [zernike_method, orb_method, neural_method, small_neural_method]
methods=[single_worst_split_method(m) for m in method_classes]
#create_databases(images,methods,image_set_name)
instantiate_databases(methods, image_set_name)

run_in_chunks3(methods, images, aberrations, chunk_size=100, weights=[],
    logdir=logdir)
print("Done")

log_files = glob("Logs/"+logdir+"/*") # these are the logs that we're loading
joined_logs = pd.concat([pd.read_csv(i) for i in log_files])

methods = list(set(joined_logs['method']))
aber = list(set(joined_logs['aberration']))
for i in methods:
    for j in aber:
        print(i + ' ' + j + ': ' + str(joined_logs[(joined_logs['method']==i) & (joined_logs['aberration']==j)]['rank'].mean()))

