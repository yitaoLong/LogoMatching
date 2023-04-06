#imports needed for the utility functions
import numpy as np
import pandas as pd
import cv2
import math
import pickle
import random
from time import perf_counter
from glob import glob
from tqdm import tqdm, tnrange, tqdm_notebook
import os
import scipy.stats as sstats
from PIL import Image, ImageOps
import torch
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation

import skimage.color
import skimage.filters

# the utility functions
 
def save_obj(obj, name ):
    # save pickle objects
    if not os.path.exists("obj"):
        os.mkdir("obj")
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    # load pickle objects
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)        
    
    
def create_databases(imgs, methods, name):
    for method in methods:
        db = method.generate_database(imgs)
        filename="db_"+method.name()+"_"+name
        save_obj(db,filename)
    return methods
def instantiate_databases(methods,name):
    for method in methods:
        filename="db_"+method.name()+"_"+name
        db = load_obj(filename)
        method.database=db
    return methods

def scast2(x,r):
        xpad=np.pad(x,[[0,0],[0,1]],mode="constant", constant_values=r)
        return xpad/np.linalg.norm(x,axis=1,keepdims=True)
def genkeys(n,d):
    r=np.random.normal(size=(n,d))
    r=r/np.linalg.norm(r,axis=1,keepdims=True)
    r=np.pad(r,[[1,0],[0,1]],mode="constant",constant_values=0)
    r[0,-1]=1
    return r
def meancolor(img,b):
    return np.floor(np.sum(img[b],(0)) / (0.01+np.sum(b))).astype(np.uint8)
def gsb(s):
    x,y=np.where(s)
    return x.min(), y.min(), x.max(), y.max()
def cropresize(img,b, bg=None):
    try:
        x1,y1,x2,y2=gsb(b)
        if bg is not None:
            img=img.copy()
            img[b!=True]=bg
        return cv2.resize(img[x1:x2,y1:y2,:],img.shape[:-1])
    except:
        return None
def maskout(img,b,bg):
    try:
        return np.where(b.reshape(img.shape[:-1]+[1]),stdl, bg if bg is not None else 0)
    except:
        return None
        
def segment_labels(img,labels,bg=None,rejection=0.0,**kwargs):
    ulabels=np.unique(labels)
    h,w=labels.shape
    total=h*w
    masks=[labels==i for i in ulabels]
    # throw away any masks that cover less than 'rejection' percent of the image
    masks=[m for m in masks if np.sum(m) >= (rejection*total)]
    segs= [cropresize(img,b,bg,**kwargs) for b in masks]
    return [s for s in segs if s is not None]
    
def segment_agglomerative(img,n_clusters=3,**kwargs):
    feats = img.reshape(-1,3)
    connectivity=grid_to_graph(*img.shape[:2])
    clustering=AgglomerativeClustering(n_clusters=n_clusters,connectivity=connectivity)
    clustering.fit(feats)
    return segment_labels(img,clustering.labels_.reshape(img.shape[:2]),**kwargs)
    
    
def color_segments(img, threshold=.98):
    # image is assumed to be HxWx3
    v=np.reshape(img,(-1,3))/255
    n=scast2(v-np.mean(v,0),0.01)
    r=genkeys(200,3)

    ids=np.argmax(n.dot(r.transpose()), axis=1)
    uids=np.unique(ids)
    w=np.array([np.sum(ids==i) for i in uids])/ids.size
    ws=np.argsort(-w)
    sids=uids[ws]
    csm=np.cumsum(w[ws])
    iids=np.array([sids[i] for i in range(1,len(sids)) if csm[i-1]<threshold])
    rids=ids.reshape(img.shape[:-1])
    return rids,iids,sids[0]


def split_image(img,method="color",center=True,hulls=False):
    # pass this an image, and it will return a list of sub-images.
    # method: determines whether to split by color or using the contour method
    # center: whether to center and resize the split parts, or to leave them in their original location in the image
    # hulls: if true use the convex hull of the contour, otherwise use the raw contour 
    masks=[]
    bgcolor=None
    if method=="color":
        rids,iids,bgid=color_segments(img)
        bgcolor=meancolor(img,rids==bgid)
        masks += [rids==i for i in iids]
    else:
        img2=prep_img(img, False)
        contours,edges=find_contours(img2)
        if hulls:
            contours=[cv2.convexHull(c) for c in contours if len(c)>2]
        mask = np.zeros_like(img2) # Create mask where white is what we want, black otherwise
        for i in range(len(contours)):
            masks += [cv2.drawContours(mask, contours, i, 1, 0)]
    if center:
        return [cropresize(img,b,bgcolor) for b in masks]
    else:
        return [maskout(img,b,bgcolor) for b in masks]        


def thresholding(img):
    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(img)

    # create a mask based on the threshold
    t = 0.8
    binary_mask = gray_image < t

    tmp = [binary_mask for _ in range(3)]

    tmp = np.stack(tmp, axis=2) * 255
    
    tmp = tmp.astype(np.uint8)
    
    return tmp

def edging(img):

    edges = canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    tmp = [edges for _ in range(3)]

    tmp = np.stack(tmp, axis=2)
    
    return tmp
        

def gray( img ):
    # create grayscale image from database image
    
    # if the image is already grayscale, just return
    if(len(np.shape(img)) < 3):
        return img
    
    if np.shape(img)[2] == 3:
        img = np.dot( img, [0.299, 0.587, 0.114] )
    else:
        img = np.dot( np.transpose( img, (1,2,0)), [0.299, 0.587, 0.114] )
    return img.astype(np.uint8)

def add_border(img, size):
    rows,cols = img.shape[:2]
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    try:
        common = int(max(set(border), key = border.count))
        array = cv2.copyMakeBorder( img,  size, size, size, size, cv2.BORDER_CONSTANT, value =  common)
    except:
        common = sstats.mode(border)[0][0] # get the most common color
        common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
        array = cv2.copyMakeBorder( img,  size, size, size, size, cv2.BORDER_CONSTANT, value =  common)

    return array.astype(np.uint8)
    
def image_preprocess(img):
    img = gray(img)
    rows,cols = img.shape[:2]
    # create grayscale image and use Canny edge detection
    cimg = canny(img)   
    
    dcimg, rows, cols, min_x, min_y, max_x, max_y = fill_in_diagonals(cimg)
    
    return rows, cols, min_x, min_y, max_x, max_y, dcimg    

def prep_img(img,grayscale=True):
    if grayscale:
        img = gray(img)
    img = cv2.medianBlur(img,3)
    img = add_border(img, 16)
    rows,cols = img.shape[:2]
    scale = (256 / max(rows,cols))
    #### resize without antialiasing
    #img = cv2.resize(img,  (int(rows * scale), int(cols * scale)))  
    ### resize with antialiasing
    img = Image.fromarray(img)
    img = ImageOps.fit(img, (int(rows * scale), int(cols * scale)), Image.ANTIALIAS)
    img = np.asarray(img)

    return img

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


## all the functions related to zernike and contour
def find_max( point_list):

    max_dist = 0.000
    second_dist = 0.000
    first_point = point_list[0]
    second_point = point_list[1]
    third_point = point_list[0]
    fourth_point = point_list[1]

    for i in range(0, len(point_list)-1):
        for j in range(i+1, len(point_list)):
            
            dist_sq = (point_list[i][0] - point_list[j][0])**2 + (point_list[i][1] - point_list[j][1])**2  
            
            if dist_sq > max_dist:
                second_dist = max_dist 
                third_point = first_point
                fourth_point = second_point
                
                max_dist = dist_sq
                first_point = point_list[i]
                second_point = point_list[j]
            
            elif dist_sq <= max_dist and dist_sq > second_dist:
                second_dist = dist_sq
                third_point = point_list[i]
                fourth_point = point_list[j]
                
    return first_point, second_point, math.sqrt(max_dist), third_point, fourth_point, math.sqrt(second_dist)   

    
# if object is a circle, find max distance which also has the closest slope to the best fit slope 
def find_tiebreaker( x, y, best_slope, max_dist ):

    first_point = (x[0], y[0])
    second_point = (x[1], y[1])
    min_slope_diff = 999999.999
    
    for i in range(0, len(x)-1):
        for j in range(i+1,len(x)+i):
        
            if j >= len(x):
                j = j % len(x)
            
            dist = math.sqrt( (x[i] - x[j])**2 + (y[i] - y[j])**2 )
            if (max_dist - dist)/max_dist <= 0.08 and (x[i] - x[j]) != 0 :
                slope_diff = abs( (y[i] - y[j])/(x[i] - x[j]) - best_slope )
                
                if slope_diff < min_slope_diff:
                    min_slope_diff = slope_diff
                    first_point = (x[i], y[i])
                    second_point = (x[j], y[j])    

    return first_point, second_point
    

# find midpoints between two points given a set of fractions  
def get_midway (p1, p2, fractions):
    
    midpts = []
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    for n, fract in enumerate(fractions):
        x = p1[0] + fract * dx 
        y = p1[1] + fract * dy 
        midpts.append( (x,y) )
            
    return midpts

# find intersection of 2 lines     
def line_intersect(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    div = det(xdiff, ydiff)
    if div == 0:
        return (-999999,-999999) 

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y    
    
# find points of intersection between normal line and all possible line segments in contour 
def find_intersect( x1, y1, x2, y2):
    
    intersect = []
    
    for i in range(0,len(x2)):
        j = i + 1
        if j == len(x2):
            j = 0
        
        # find point of intersection
        point = line_intersect( ((x1[0], y1[0]), (x1[-1], y1[-1])), ((x2[i], y2[i]), (x2[j], y2[j])) )
        
        # check that the point lies on the contour 
        dotproduct = (point[0] - x2[i]) * (x2[j] - x2[i]) + (point[1] - y2[i])*(y2[j] - y2[i])
        squaredlength = (x2[j] - x2[i])**2 + (y2[j] - y2[i])**2
        
        if dotproduct >= 0 and dotproduct <= squaredlength and point not in intersect:
            intersect.append( point )

    return intersect

def split_contour(x_cnt, y_cnt, x_mid, y_mid):
    
    if x_cnt > x_mid:
        return 0
    elif x_cnt == x_mid and y_cnt > y_mid:
        return 0
    else:
        return 0.5
    
def frange(start, stop, step):
    i = start
    while i < stop:
        if i == int(i):
            i = int(i)
        yield i
        i += step


def canny(img):
    
    threshhold = [20,60,120, 200]
    pixel_count = []
    
    for t in range(0,len(threshhold)-1):
    
        edges1 = cv2.Canny(img,threshhold[t],threshhold[t+1])
        pixel_count_temp = np.sum(np.array(edges1) >= 200)
        pixel_count.append(pixel_count_temp)
        
    t_max = pixel_count.index(max(pixel_count))
    edges1 = cv2.Canny(img,threshhold[t_max],threshhold[t_max+1])    
    return edges1

def fill_in_diagonals(img):
    rows,cols = img.shape[:2]
    min_x = cols 
    min_y = rows
    max_x = 0 
    max_y = 0 
    # fill in any diagonals in the edges 
    for i in range(0, rows):
        for j in range(0, cols):
            if  i >= 1 and i <= rows - 2 and j >= 1 and j <= cols - 2: 
                if img[i,j] == 255 and img[i,j-1] == 0 and img[i+1,j] == 0 and img[i+1,j-1] == 255:
                    img[i+1,j] = 255 
                elif img[i,j] == 255 and img[i,j+1] == 0 and img[i+1,j] == 0 and img[i+1,j+1] == 255:
                    img[i+1,j] = 255 
            
            if img[i,j] == 255:
                if j > max_x:
                    max_x = j 
                if j < min_x:
                    min_x = j
                if i > max_y:
                    max_y = i 
                if i < min_y: 
                    min_y = i
    return img, rows, cols, min_x, min_y, max_x, max_y

def edge_detect( img ):
    edges1 = canny(img)

    edges2, rows, cols, min_x, min_y, max_x, max_y = fill_in_diagonals(edges1)
                    
    edges2 = edges2[min_y:max_y+1, min_x:max_x+1]
    row_dist = max_y+1-min_y
    col_dist = max_x+1-min_x
    scale = int(1000 / max(row_dist,col_dist))
    try:
        edges2 = cv2.resize(edges2, dsize=(row_dist*scale, col_dist*scale), interpolation=cv2.INTER_CUBIC)                
    except:
        edges2 = edges1
        
    return edges1, edges2 

def find_contours(img, n = 20):

    gamma_list = [1]
    perimeter_list = []
    
    for g in range(0,len(gamma_list)):

        img_gamma = adjust_gamma(img, gamma_list[g])
    
        # create grayscale image and use Canny edge detection
        edges_gamma, edges2_gamma = edge_detect(img_gamma) 
        
        # use ellipse dilation to fill the gaps in countours  
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        #edges_gamma = cv2.dilate(edges_gamma, kernel)
        #edges2_gamma = cv2.dilate(edges2_gamma, kernel)
        
        # find contours from edge image    
        try:
            contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        except:
            image, contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            
        # use 10 biggest countours      
        contours_gamma = sorted(contours_gamma, key = lambda x:cv2.contourArea(x), reverse = True)[:n]  
        total_perimeter_gamma = 0
        for n, contour_g in enumerate(contours_gamma):
            total_perimeter_gamma += cv2.arcLength(contour_g,False)   

        perimeter_list.append(total_perimeter_gamma)  
        
    g_max = perimeter_list.index(max(perimeter_list))   
    img_gamma = adjust_gamma(img, gamma_list[g_max]) 
    edges_gamma, edges2_gamma = edge_detect(img_gamma)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #edges_gamma = cv2.dilate(edges_gamma, kernel)
    try:
        contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    except:
        image, contours_gamma, hier = cv2.findContours(edges_gamma,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            
    contours_gamma = sorted(contours_gamma, key = lambda x:cv2.contourArea(x), reverse = True)[:n]  
      
    return contours_gamma, edges2_gamma


# functions to save and load databases
# the advantage of structuring all the methods in a similar way is that we can write looping code like this
def generate_databases(imgs, method_classes, name):
    for method_c in method_classes: # for each method class
        method = method_c() # create the method by instancing the class
        db = method.generate_database(imgs) # generate the database from the images
        filename = "db_"+method.__class__.__name__+"_"+name # this is what the database file should be called
        save_obj(db, filename) # save it
        
def load_databases(method_classes, name):
    loaded_methods = [] # list of loaded methods to return
    for method_c in method_classes: # for each method class
        filename = "db_"+method_c.__name__+"_"+name # this is what the database file should be called
        db = load_obj(filename) # load the database file
        method = method_c(db) # construct the instance of the method class using the database
        loaded_methods.append(method) # add it to the list of methods to return
    return loaded_methods


def plot_results (img, results, images, result_index = 1):
    # img is the original image
    # results is a list of (index, score1, score2...) tuples in descending order of match closeness, eg [(1, 100), (0, 77.7), (2, 15.6)]    
    # images is the set of images we're checking against
    
    # create plot of original image and best matches 
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(32, 32),sharex=False, sharey=False)
    ( (ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12) ) = ax
    result_cells = [ax3, ax4, ax5, ax6, ax9, ax10, ax11, ax12]

    # this part is just so that we can make the original image bigger than the others
    gs = ax[1,2].get_gridspec()
    ax1.remove()
    ax2.remove()
    ax7.remove()
    ax8.remove()
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.imshow(gray(img), cmap=plt.cm.gray)
    ax1.set_title('Query Image', fontsize=20, y = 1.0)

    # this shows the result images
    for c_inx in range(len(result_cells)):
        if c_inx >= len(images):
            break
        result_cells[c_inx].imshow(gray(images[results[c_inx][0]]), cmap=plt.cm.gray)
        result_cells[c_inx].set_xlim([0,32])
        result_cells[c_inx].set_ylim([32,0])
        result_cells[c_inx].set_title('match score: ' + '%.1f'%(results[c_inx][result_index]), fontsize=20, y = 1.0)
    
    # maximize the window and display plots 
    fig.tight_layout()
    plt.show()

def test_combined(methods, weights = []):
    # this version of test_combined should be able to take any number of methods
    # if the weights are not specified, will weigh everything equally
    match_combined = [] # this will have an entry [index, m1_score, m2_score,..., combined_score] for each image
    num_imgs = len(methods[0])
    num_methods = len(methods)
    for idx, entry in enumerate(zip(*methods)): # *methods 'unpacks' the list, basically stripping the outside pair of brackets
        # for each image we build up an entry [index, m1_score, m2_score,..., combined_score]
        
        # score serves as an accumulator for the score, we'll keep adding the score of this image
        # for each method to this, and then store the final result at the end of the entry
        score = 0 
        tmp = [idx] # this will be the entry, starts off with the image index
        for m_idx, method_score in enumerate(entry): # then for each method,
            tmp.append(method_score[1]) # we add the individual score to the entry
            score += method_score[1] * (weights[m_idx] if m_idx < len(weights) else 1) # and accumulate the weighted score
        tmp.append(score) # after we look at each method we add the combined weighted score to the entry
        match_combined.append(tmp) # and then we add the entry to the matched list
    match_combined = sorted(match_combined, key = lambda tup: tup[-1], reverse = True ) # sort by combined score 
    return match_combined
    
    
def chunks(x, n=10):
    for i in range(0, len(x), n):
        yield x[i:i+n]

def run_in_chunks(methods, images, aberrations, weights=[], chunk_size=100):
    candidates = [i for i in range(len(images))]
    random.shuffle(candidates)
    for chunk_num, candidate_chunk in enumerate(chunks(candidates, chunk_size)):
        print("Chunk: "+str(chunk_num+1))
        for aber in aberrations:
            run(methods, images, aber=aber, candidates=candidate_chunk, weights=weights)
    
   
def run4(methods, images, positive_num, negative_num, aber=None, candidates=None, weights=[], logdir = "Logs"):
    candidates = candidates or range(len(images))
    results = {} # this dictionary will be used to collect results to log
    scores = {} # this dictionary will collect information to run the logistic regression on
    imgs_so_far = 0 # just to keep track of progress
    for img_idx in candidates: # run through all the candidates
        if imgs_so_far % 10 == 0:
            print(imgs_so_far)
        imgs_so_far += 1
        img = images[img_idx]
        if img is None: # if on the odd chance some image is missing, skip it
            continue
        query_image = aber(img) if aber is not None else img
        method_lists = [] # we're going to gather the query lists of each of the methods here so we can combine them later
        for method_idx, method in enumerate(methods): # try each of the methods
            start_time = perf_counter() # this is a timestamp of when we start the method
            xl = method.run_query(query_image, candidates=candidates) # get the scores for all the images in the method's database
            method_lists.append(xl) # store the unsorted version for easy combining
            xl = sorted(xl, key = lambda y: y[1], reverse=True) # sort the results by score
            time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
            
#            positive sample
            pos_img = np.concatenate((query_image, img), axis=0)
            pos_img = Image.fromarray(pos_img)
            positive_num += 1
            pos_img.save('./train/pos/'+ str(positive_num)  +'.png')
#            negative sample
            if xl[0][0] == img_idx:
                neg_img = np.concatenate((query_image, images[xl[1][0]]), axis=0)
            else:
                neg_img = np.concatenate((query_image, images[xl[0][0]]), axis=0)
            neg_img = Image.fromarray(neg_img)
            negative_num += 1
            neg_img.save('./train/neg/' + str(negative_num) + '.png')
    return positive_num, negative_num
            

def run_in_chunks4(methods, images, aberrations, weights=[], chunk_size=100, logdir = "Logs"):
    positive_num = 0
    negative_num = 0
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    candidates = [i for i in range(len(images))]
    random.Random(1).shuffle(candidates)
    for chunk_num, candidate_chunk in enumerate(chunks(candidates, chunk_size)):
        print("Chunk: "+str(chunk_num+1))
        for aber in aberrations:
            a, b = run4(methods, images, positive_num, negative_num, aber=aber, candidates=candidate_chunk, weights=weights, logdir=logdir)
            positive_num = a
            negative_num = b
    

def run5(methods, images, feature_extractor, model, aber=None, candidates=None, weights=[], logdir = "Logs"):
    # this version of run builds a dictionary containing stats for each (aberrated) image, rather than just collecting stats
    # in aggregate. This will then be piped into a dataframe so we can get any statistics we want.
    candidates = candidates or range(len(images))
    
    rank_count = 0
    for img_idx in candidates: # run through all the candidates
        img = images[img_idx]
        if img is None: # if on the odd chance some image is missing, skip it
            continue
        query_image = aber(img) if aber is not None else img
        method_lists = [] # we're going to gather the query lists of each of the methods here so we can combine them later
        for method_idx, method in enumerate(methods): # try each of the methods
            start_time = perf_counter() # this is a timestamp of when we start the method
            xl = []

            encoding = []
            for image_index in candidates:
                combined_img = np.concatenate((query_image, images[image_index]), axis=0)
                encoding.append(Image.fromarray(combined_img).convert('RGB'))

            encoding = feature_extractor(encoding, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoding)
                res_score = outputs.logits[:, 1]
            
            for iii in range(len(res_score)):
                xl.append((candidates[iii], res_score[iii]))
            
            method_lists.append(xl) # store the unsorted version for easy combining
            xl = sorted(xl, key = lambda y: y[1], reverse=True) # sort the results by score
            time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
            score = 0 # once we find the input image, we're gonna put the score of it in here
            rank = 0 # we're gonna count up to the rank of the input image
            for hit in xl: # go through all the scores
                rank += 1
                if hit[0] == img_idx: # stop when we've found the input image
                    score = hit[1]
                    break
            rank_count += rank
    return rank_count / len(candidates)
    

def run_in_chunks5(methods, images, aberrations, feature_extractor, model, weights=[], chunk_size=100, logdir = "Logs"):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    candidates = [i for i in range(len(images))]
    random.Random(1).shuffle(candidates)
    for aber in aberrations:
        res_count = 0
        for chunk_num, candidate_chunk in enumerate(chunks(candidates, chunk_size)):
            res_count += run5(methods, images, feature_extractor, model, aber=aber, candidates=candidate_chunk, weights=weights, logdir=logdir)
        print(aber.__name__ if aber is not None else 'ab_id')
        print(res_count / math.ceil(len(images) / chunk_size))




def run9(methods, images, positive_num, negative_num, aber=None, candidates=None, weights=[], logdir = "Logs"):
    # this version of run builds a dictionary containing stats for each (aberrated) image, rather than just collecting stats
    # in aggregate. This will then be piped into a dataframe so we can get any statistics we want.
    candidates = candidates or range(len(images))
    results = {} # this dictionary will be used to collect results to log
    scores = {} # this dictionary will collect information to run the logistic regression on
    imgs_so_far = 0 # just to keep track of progress
    for img_idx in candidates: # run through all the candidates
        if imgs_so_far % 10 == 0:
            print(imgs_so_far)
        imgs_so_far += 1
        img = images[img_idx]
        if img is None: # if on the odd chance some image is missing, skip it
            continue
        query_image = aber(img) if aber is not None else img
        method_lists = [] # we're going to gather the query lists of each of the methods here so we can combine them later
        for method_idx, method in enumerate(methods): # try each of the methods
            start_time = perf_counter() # this is a timestamp of when we start the method
            xl = method.run_query(query_image, candidates=candidates) # get the scores for all the images in the method's database
            method_lists.append(xl) # store the unsorted version for easy combining
            xl = sorted(xl, key = lambda y: y[1], reverse=True) # sort the results by score
            time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
            
#            positive sample
            query_grey = thresholding(query_image)
            query_edge = edging(query_image)
            img_grey = thresholding(img)
            img_edge = edging(img)
            pos_img = np.concatenate((query_image, query_grey, query_edge, img, img_grey, img_edge), axis=0)
            pos_img = Image.fromarray(pos_img)
            positive_num += 1
            pos_img.save('./train/pos/'+ str(positive_num)  +'.png')
#            negative sample
            if random.random() >= 0.5:
                random_index = 0
                while True:
                    random_index = random.randint(0, len(images)-1)
                    if random_index != img_idx:
                        break
                neg_img_grey = thresholding(images[random_index])
                neg_img_edge = edging(images[random_index])
                neg_img = np.concatenate((query_image, query_grey, query_edge, images[random_index], neg_img_grey, neg_img_edge), axis=0)
            else:
                if xl[0][0] == img_idx:
                    neg_img_grey = thresholding(images[xl[1][0]])
                    neg_img_edge = edging(images[xl[1][0]])
                    neg_img = np.concatenate((query_image, query_grey, query_edge, images[xl[1][0]], neg_img_grey, neg_img_edge), axis=0)
                else:
                    neg_img_grey = thresholding(images[xl[0][0]])
                    neg_img_edge = edging(images[xl[0][0]])
                    neg_img = np.concatenate((query_image, query_grey, query_edge, images[xl[0][0]], neg_img_grey, neg_img_edge), axis=0)
            neg_img = Image.fromarray(neg_img)
            negative_num += 1
            neg_img.save('./train/neg/' + str(negative_num) + '.png')
    return positive_num, negative_num


def run_in_chunks9(methods, images, aberrations, weights=[], chunk_size=100, logdir = "Logs"):
    positive_num = 0
    negative_num = 0
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    candidates = [i for i in range(len(images))]
    random.Random(1).shuffle(candidates)
    for chunk_num, candidate_chunk in enumerate(chunks(candidates, chunk_size)):
        print("Chunk: "+str(chunk_num+1))
        for aber in aberrations:
            a, b = run9(methods, images, positive_num, negative_num, aber=aber, candidates=candidate_chunk, weights=weights, logdir=logdir)
            positive_num = a
            negative_num = b

def run10(methods, images, feature_extractor, model, aber=None, candidates=None, weights=[], logdir = "Logs"):
    # this version of run builds a dictionary containing stats for each (aberrated) image, rather than just collecting stats
    # in aggregate. This will then be piped into a dataframe so we can get any statistics we want.
    candidates = candidates or range(len(images))
    
    rank_count = 0
    imgs_so_far = 0
    for img_idx in candidates: # run through all the candidates
#        if imgs_so_far % 10 == 0:
#            print(imgs_so_far)
        imgs_so_far += 1
        img = images[img_idx]
        if img is None: # if on the odd chance some image is missing, skip it
            continue
        query_image = aber(img) if aber is not None else img
        method_lists = [] # we're going to gather the query lists of each of the methods here so we can combine them later
        for method_idx, method in enumerate(methods): # try each of the methods
            start_time = perf_counter() # this is a timestamp of when we start the method
            xl = []

            encoding = []
            for image_index in candidates:
                query_grey = thresholding(query_image)
                query_edge = edging(query_image)
                img_grey = thresholding(images[image_index])
                img_edge = edging(images[image_index])
                combined_img = np.concatenate((query_image, query_grey, query_edge, images[image_index], img_grey, img_edge), axis=0)
                encoding.append(Image.fromarray(combined_img))

            encoding = feature_extractor(encoding, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoding)
                res_score = outputs.logits[:, 1]
            
            for iii in range(len(res_score)):
                xl.append((candidates[iii], res_score[iii]))
            
            method_lists.append(xl) # store the unsorted version for easy combining
            xl = sorted(xl, key = lambda y: y[1], reverse=True) # sort the results by score
            time_elapsed = perf_counter() - start_time # ending timestamp minus start is the elapsed time
            score = 0 # once we find the input image, we're gonna put the score of it in here
            rank = 0 # we're gonna count up to the rank of the input image
            for hit in xl: # go through all the scores
                rank += 1
                if hit[0] == img_idx: # stop when we've found the input image
                    score = hit[1]
                    break
            rank_count += rank
    return rank_count / len(candidates)
    

def run_in_chunks10(methods, images, aberrations, feature_extractor, model, weights=[], chunk_size=100, logdir = "Logs"):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    candidates = [i for i in range(len(images))]
    random.Random(1).shuffle(candidates)
    for aber in aberrations:
        res_count = 0
        for chunk_num, candidate_chunk in enumerate(chunks(candidates, chunk_size)):
            res_count += run10(methods, images, feature_extractor, model, aber=aber, candidates=candidate_chunk, weights=weights, logdir=logdir)
        print(aber.__name__ if aber is not None else 'ab_id')
        print(res_count / math.ceil(len(images) / chunk_size))
