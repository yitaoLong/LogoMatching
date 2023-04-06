import numpy as np
import cv2
from scipy import stats as sstats
import random
from PIL import Image
from skimage.feature import canny
from skimage.filters import sobel
import skimage.color
import skimage.filters

def ab_id(img):
    return img

def ab_rotate(img, border=None, borderMode=cv2.BORDER_REPLICATE):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),random.randint(0,360),1)
    if border is not None:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=borderMode)
    return img

def ab_affine(img, border=None):
    # cv2.getAffineTransform takes 3 input points and 3 output points, and returns the affine transformation matrix mapping the input points to the output.
    rows,cols,_ = img.shape
    pts1 = np.float32([[0,0],[0,30],[30,30]])
    pts2 = np.float32([[random.randint(0,10),random.randint(0,10)],[random.randint(0,10),random.randint(20,30)],[random.randint(20,30),random.randint(20,30)]])
    M = cv2.getAffineTransform(pts1,pts2)
    if border is not None:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_scale(img, border=None):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),0,random.uniform(.8, 1.25))
    if border is not None:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    else:
        img = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REPLICATE)
    return img

def ab_flip(img):
    img = cv2.flip(img, random.randint(0,1))
    return img

basic_geometric = [ab_translate, ab_rotate, ab_affine, ab_scale, ab_flip]

# the next two are the basic drawing transformations
def ab_line(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.line(img,(random.randint(15,25),random.randint(15,25)),(random.randint(15,25),random.randint(15,25)),(0,300,300),random.randint(1,3))
    return img

def ab_circle(img):
    img = img.copy() # there's a bug in opencv when drawing lines, this is the workaround
    cv2.circle(img, (random.randint(12,20),random.randint(12,20)), random.randint(2,4), (0,300,300), random.randint(-3, 3))
    return img

basic_drawing = [ab_line, ab_circle]

# border_color gets the most common color along the border to use when applying a geometric transformation
def border_color(img):
    rows,cols,_ = img.shape
    firstcol = img[:,0]
    lastcol = img[:,cols-1]
    firstrow = img[0,:]
    lastrow = img[rows-1,:]
    border = list(np.concatenate((firstcol,lastcol,firstrow,lastrow)))
    common = sstats.mode(border)[0][0] # get the most common color
    common = tuple(map(int, common)) # convert to a tuple of ints to pass as color
    return common

def ab_rotate_border(img):
    border = border_color(img)
    img = ab_rotate(img, border)
    return img

def ab_translate_border(img):
    border = border_color(img)
    img = ab_translate(img, border)
    return img

def ab_scale_border(img):
    border = border_color(img)
    img = ab_scale(img, border)
    return img

def ab_affine_border(img):
    border = border_color(img)
    img = ab_affine(img, border)
    return img

border_geometric = [ab_rotate_border, ab_translate_border, ab_scale_border, ab_affine_border]
all_geometric = border_geometric+[ab_flip]

# this applies multiple drawings
def ab_draw(img):
    num_drawings = random.randint(1,3)
    for i in range(num_drawings):
        img = np.random.choice(basic_drawing)(img)
    return img

def ab_repeated(img):
    bg = Image.fromarray(img)

    # The width and height of the background tile
    bg_w, bg_h = bg.size

    # Creates a new empty image, RGB mode, and size 96 by 96
    new_im = Image.new('RGB', (96,96))

    # The width and height of the new image
    w, h = new_im.size

    # Iterate through a grid, to place the background tile
    for i in range(0, w, bg_w):
        for j in range(0, h, bg_h):
            # Change brightness of the images, just to emphasise they are unique copies
            bg = Image.eval(bg, lambda x: x+(i+j)/96)

            #paste the image at location i, j:
            new_im.paste(bg, (i, j))

    img = np.array(new_im.resize((32, 32)))
    
    t0 = img[:,:,0]
    t1 = img[:,:,1]
    t2 = img[:,:,2]
    img[:,:,0] = t1
    img[:,:,1] = t2
    img[:,:,2] = t0
    return img

def ab_thresholding(img):
    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(img)

    # create a mask based on the threshold
    t = 0.8
    binary_mask = gray_image < t

    tmp = [binary_mask for _ in range(3)]

    tmp = np.stack(tmp, axis=2) * 255
    
    tmp = tmp.astype(np.uint8)
    
    return tmp

def ab_edge(img):
    edges = canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    tmp = [edges for _ in range(3)]

    tmp = np.stack(tmp, axis=2)
    
    return tmp
    
all_advanced = [ab_draw]

aberrations_1 = [ab_id]+basic_drawing+all_geometric+all_advanced+[ab_repeated]
aberrations_2 = aberrations_1 + [ab_edge, ab_thresholding]

# aberrations function
def get_ab(img, n=None):
    # This function takes an input image (32 by 32) and applies a random aberration to it, either some affine transformation or an occlusion N times.
    # Note that the input image is mutated.
    ab = None
    if n is not None and n >= 0 and n < len(aberrations): # if we want a specific ab then give it
        ab = aberrations[n]
    else:
        ab = np.random.choice(aberrations) # otherwise pick at random
    img = ab(img)
    return img, ab
