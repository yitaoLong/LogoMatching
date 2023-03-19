# these are the imports needed to run the methods
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import math
import mahotas
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.preprocessing import normalize

from libseg.icon_util import *

class method_base(object):
    # base class that all methods inherit from
    # this is just to avoid having to rewrite all the stuff that is the same between
    # the various methods
    
    def name(self):
        return "Base"

    # The database stores the image descriptors for all the known images,
    # indexed by the image index.
    # Either initialize as an empty dict or use the one passed in
    def __init__(self, database = None):
        self.database = database or {}

    # Take in an image and return its descriptor calculated by the method.
    # Each method will override this.
    def create_query(self, img, **kwargs):
        return 1

    # Take in two descriptors and say how similar they are.
    # Each method will override this.
    def compare_queries(self, v1, v2, **kwargs):
        return 1

    # If the method needs to normalize its results somehow, do that by
    # overriding this function.
    def normalize_results(self, results, **kwargs):
        return results

    # The last three functions should be the same for all methods.

    # Take an image descriptor, and optionally a set of indices for images it's likely to be
    # and return a list of (index, similarity) pairs for each candidate image,
    # or for every known image if candidates are not given.
    def test_query(self, img_vec, candidates=None, **kwargs):
        candidates = candidates or self.database.keys()
        results = [(img_index, self.compare_queries(img_vec, self.database[img_index], **kwargs)) for img_index in candidates]
        return self.normalize_results(results)

    # Just a convenience function to get an image descriptor and run the comparison.
    def run_query(self, img, **kwargs):
#       if flag is 1, means the whole image
        img_query = self.create_query(img, flag=1, **kwargs) # create the image descriptor
        results = self.test_query(img_query, **kwargs) # compare against all the images in the database
        return results

    # Make and store image descriptors for every given image.
    def generate_database(self, images, **kwargs):
        self.database  = {}
        for img_index in range(len(images)):
            if img_index%50 == 0:
                print(img_index)
            img = images[img_index]
            self.database[img_index] = self.create_query(img, **kwargs)
        return self.database


class split_cluster(method_base):
    def name(self):
        return "SplitCluster_" + str(self.split_params)+ "_" + self.method.name()
        
    def __init__(self, method_class, database = None, **kwargs):
        self.database = database or {}
        self.method=method_class()
        self.split_params=kwargs
    
    def create_query(self, img, flag=0, cache=None, **kwargs):
        if flag is 1:
            return [np.array(self.method.create_query(img, **kwargs))]
        try:
            res = []
            splits = []
            if cache is not None:
                splits=cache
            else:
                splits=segment_agglomerative(img, rejection=0.0)
            for s in splits:
                res.append(self.method.create_query(s, **kwargs))
            return res
        except:
            return None

    def compare_queries(self, v1s, v2s, s3, **kwargs):
        if len(v1s)==0 or len(v2s)==0:
            return 0
        res = []
        for v1 in v1s:
            for v2 in v2s:
                res.append(self.method.compare_queries(v1,v2,**kwargs))
        res = np.array(res)
        tmp = np.max(res)
        #print(tmp.shape)
        return tmp
        return np.max(res)


class split_method(method_base):
    # wrapper around a method to enable it to work with the constituent shapes of an image
    def name(self):
        return "Split_" + str(self.split_params)+ "_" + self.method.name()

    
    
    # The database stores the image descriptors for all the known images,
    # indexed by the image index.
    # Either initialize as an empty dict or use the one passed in
    def __init__(self, method_class, database = None, **kwargs):
        self.database = database or {}
        self.method=method_class()
        self.split_params=kwargs
        
    def create_query(self, img, flag=0, cache=None, **kwargs):
        if flag is 1:
            return [np.array(self.method.create_query(img, **kwargs))]
        try:
            res = []
            splits = []
            if cache is not None:
                splits=cache
            else:
                splits=split_image(img, **self.split_params)
            for s in splits:
                res.append(self.method.create_query(s, **kwargs))
            return res
        except:
            return None

        
    def compare_queries(self, v1s, v2s, **kwargs):
        if len(v1s)==0 or len(v2s)==0:
            return 0
        res = []
        for v1 in v1s:
            for v2 in v2s:
                res.append(self.method.compare_queries(v1,v2,**kwargs))
        res = np.array(res)
        return np.sum(res)/res.shape(0)
    
    def comparison_matrix(self,v1s,v2s,**kwargs):
        if len(v1s)==0 or len(v2s)==0:
            return [[0]]
        res = []
        for v1 in v1s:
            r1 = []
            for v2 in v2s:
                r1.append(self.method.compare_queries(v1,v2,**kwargs))
            res.append(r1)
        return res

    def normalize_results(self, results, **kwargs):
        return self.method.normalize_results(results)
    
    
class single_best_split_method(split_method):
    # change comparison of split methods to take the single best match from the bunch
    def name(self):
        return "Single_Best_Split_" + str(self.split_params)+ "_" + self.method.name()
        
    def compare_queries(self, v1s, v2s, **kwargs):
        if len(v1s)==0 or len(v2s)==0:
            return 0
        res = []
        for v1 in v1s:
            for v2 in v2s:
                res.append(self.method.compare_queries(v1,v2,**kwargs))
        res = np.array(res)
        tmp = np.max(res)
        #print(tmp.shape)
        return tmp
        return np.max(res)
    
class single_worst_split_method(split_method):
    # change comparison of split methods to take the single best match from the bunch
    def name(self):
        return "Single_Worst_Split_" + str(self.split_params)+ "_" + self.method.name()
        
    def compare_queries(self, v1s, v2s, **kwargs):
        if len(v1s)==0 or len(v2s)==0:
            return 0
        res = []
        for v1 in v1s:
            for v2 in v2s:
                res.append(self.method.compare_queries(v1,v2,**kwargs))
        res = np.array(res)
        tmp = np.min(res)
        #print(tmp.shape)
        return tmp
        return np.min(res)
    
class symmetric_best_split_method(split_method):
    # change comparison of split methods to take the single best match from the bunch
    def name(self):
        return "Symmetric_Best_Split_" + str(self.split_params)+ "_" + self.method.name()
        
    def compare_queries(self, v1s, v2s, **kwargs):
        if len(v1s)==0 or len(v2s)==0:
            return 0
        res = []
        for v1 in v1s:
            r1=[]
            for v2 in v2s:
                r1.append(self.method.compare_queries(v1,v2,**kwargs))
            res.append(r1)
        res = np.array(res)*10
        r2 = (scipy.special.softmax(-res,axis=1) + scipy.special.softmax(-res,axis=0)) * res
        r3 = np.sum(res)
        #print(r3.shape)
        return r3

    
timm_names=['vit_small_r26_s32_224',
            'cspdarknet53',
            'tf_efficientnetv2_l',
            'resnet50',
            'ecaresnetlight',
            'mixnet_s',
            'res2next50',
            'wide_resnet50_2']
class timm_method(method_base):
    def __init__(self, database = None, modelname=timm_names[0], cuda=False):
        self.model = timm.create_model(modelname,pretrained=True,num_classes=0)
        if cuda:
            self.model = self.model.cuda()
        config = resolve_data_config({},model=self.model)
        self.transform = create_transform(**config)
        self.topil=transforms.ToPILImage()
        self.nameval = 'Timm_'+modelname
        self.cuda=cuda
    def name(self):
        return self.nameval
    def create_query(self,img,**kwargs):
        try:
            timg=self.topil(img)
            timg=self.transform(timg).unsqueeze(0)
            if self.cuda:
                timg=timg.cuda()
            return self.model(timg).squeeze().detach().numpy()
        except:
            return None
    def compare_queries(self,v1,v2,**kwargs):
        try:
            return np.dot(v1,v2)
        except:
            return -1.0
# neural
class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
    
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        return self.stn(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features,
                                self.out_features,
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x

class PreModel(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        
        #PRETRAINED MODEL
        self.pretrained = models.resnet18(pretrained=True)
        
        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
#         add one single spatial transformer network
        self.pretrained.maxpool = nn.Sequential(Identity(), SpatialTransformerNetwork())
        
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        
        self.projector = ProjectionHead(512, 512, 128)

    def forward(self,x):
        out = self.pretrained(x)
        return out

class stn_neural_method(method_base):
    def name(self):
        return 'stn_neural'
            
    model = PreModel('resnet18')
    model.load_state_dict(torch.load('./models/SimCLR_OUR_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_10_260621.pt', map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    
    def create_query(self, img, **kwargs):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        pillize = transforms.ToPILImage()
        loader = transforms.Compose([pillize, to_tensor, normalize])
        l_img = loader(img)
        t_img = Variable(l_img).unsqueeze(0)
        f_vec = self.model(t_img).squeeze()
        n_vec = f_vec.detach().numpy()
        n_vec = n_vec / np.linalg.norm(n_vec)
        return n_vec

    def compare_queries(self, v1, v2, **kwargs):
        cos = np.dot(v1, v2)
        sim = (cos + 1) * 50
        return sim


class neural_method(method_base):
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    #strip the final layer to get a feature vector
    model = nn.Sequential(*list(model.children())[:-1])  
    # Set model to evaluation mode
    model.eval()
    
    def name(self):
        return "Neural"

    def create_query(self, img, **kwargs):
        try:
            scaler = transforms.Resize((224, 224))
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            pillize = transforms.ToPILImage()
            loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
            l_img = loader(img)
            t_img = Variable(l_img).unsqueeze(0)
            f_vec = self.model(t_img).squeeze()
            n_vec = f_vec.detach().numpy()
            n_vec = n_vec / np.linalg.norm(n_vec)
            return n_vec
        except:
            return None

    def compare_queries(self, v1, v2, **kwargs):
        try:
            cos = np.dot(v1, v2)
            sim = (cos + 1) * 50
            return sim.item()
        except:
            return 0
    
class euclidean_neural_method(method_base):
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    #strip the final layer to get a feature vector
    model = nn.Sequential(*list(model.children())[:-1])  
    # Set model to evaluation mode
    model.eval()
    
    def name(self):
        return "Euclidean_Neural"
    
    def create_query(self, img, **kwargs):
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        pillize = transforms.ToPILImage()
        loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
        l_img = loader(img)
        t_img = Variable(l_img).unsqueeze(0)
        f_vec = self.model(t_img).squeeze()
        n_vec = f_vec.detach().numpy()
        return n_vec
    
    def compare_queries(self, v1, v2, **kwargs):
        sim = 100/(np.linalg.norm(v1-v2)+1)
        return sim
    
#class trained_neural_method(euclidean_neural_method):
#    model = models.resnet18()
#    model.fc = nn.Linear(512,8)
#    model.load_state_dict(torch.load("models/EmbeddingIconResnet.pt"))
#    model.eval()

class small_neural_method(method_base):
    # this is the definition of the custom neural network
    def name(self):
        return "Small_Neural"
    class IconEmbeddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.convnet = nn.Sequential(
                nn.Conv2d(3, 32, 5),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, 5),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2))
            self.fc = nn.Sequential(
                nn.Linear(64 * 4 * 4, 256),
                nn.PReLU(),
                nn.Linear(256, 256),
                nn.PReLU(),
                nn.Linear(256, 8)
                )
        def forward(self, x):
            output = self.convnet(x)
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
            return output

        def get_embedding(self, x):
            return self.forward(x)
        
    model = IconEmbeddingNet()
    model.load_state_dict(torch.load("models/IconEmbeddingNet.pt"))
    model.eval()
    
    def create_query(self, img, **kwargs):
        try:
            scaler = transforms.Resize((28, 28))
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            pillize = transforms.ToPILImage()
            loader = transforms.Compose([pillize, scaler, to_tensor, normalize])
            l_img = loader(img)
            t_img = Variable(l_img).unsqueeze(0)
            f_vec = self.model(t_img).squeeze()
            n_vec = f_vec.detach().numpy()
            return n_vec
        except:
            return None

    def compare_queries(self, v1, v2, **kwargs):
        try:
            sim = 100.0/(np.linalg.norm(v1-v2)+1)
            return sim
        except:
            return 0
    

# orb
class orb_method(method_base):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def name(self):
        return "Orb"

    def create_query(self, img, **kwargs):
        try:
            img = gray(img)
            img = cv2.resize(img,None, fx=13, fy=13, interpolation=cv2.INTER_AREA)

            # find the keypoints and descriptors with orb
            kp1, des1 = self.orb.detectAndCompute(img,None)
            if len(kp1) < 2:
                des1 = None
            return des1
        except:
            return None

    def compare_queries(self, img_kp1, img_kp2, **kwargs):
        if img_kp1 is None or img_kp2 is None:
            return 0
        try:
            matches = self.matcher.match(img_kp1,img_kp2)
            score = len(matches)
            return score
        except:
            return 0

    def normalize_results(self, results, **kwargs):
        # find the result with the highest score, then divide everything
        # accordingly so that score is 100
        best = 1
        for i in results:
            if i[1] > best:
                best = i[1]
        results = [(x[0],round(100 * x[1]/best)) for x in results]
        return results


class zernike_method(method_base):
    def name(self):
        return "Zernike"
    def create_query(self, img, **kwargs):
        try:
            _, _, min_x, min_y, max_x, max_y, edges1 = image_preprocess(img) 
            ## create zernike vector 
            edges2 = edges1[min_y:max_y+1, min_x:max_x+1]
            edges2 = cv2.resize(edges2, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            zernike = mahotas.features.zernike_moments(edges2, 16)  
            return normalize(zernike[:,np.newaxis], axis=0).ravel()
        except:
            return None

    def compare_queries(self, x,y, **kwargs):
        try:
            dot_prod = sum(i[0] * i[1] for i in zip(x, y))
            return 50.0 * (dot_prod + 1.0) 
        except:
            return 0


class contour_method(method_base):
    
    #fractions = [.1,.2,.3,.4,.5,.6,.7,.8,.9]    
    fractions = [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]

    def name(self):
        return "Contour"
    
    def create_subimages(self, img, fractions=None, **kwargs): 

        rows, cols = img.shape[:2]  
        if max(rows,cols) < 64:
            img = add_border(img, 16)    
            rows,cols = img.shape[:2]
            if max(rows,cols) < 256:
                scale = 4
            else:
                scale = 1
            img = Image.fromarray(img)
            img = ImageOps.fit(img, (int(rows * scale), int(cols * scale)), Image.ANTIALIAS)
            img = np.asarray(img)

        contours, edges = find_contours(img)
        x_list=[]
        y_list=[]
        for n, cnt in enumerate(contours):
        
            x_coord = cnt[:,0, 0]
            y_coord = cnt[:,0, 1] 
            x_list.append(x_coord)
            y_list.append(y_coord)

        return x_list, y_list
    
    def create_query(self, img, fractions=None, **kwargs):

        rows, cols = img.shape[:2]
        if rows < 130:
            img = prep_img(img)

        rows, cols = img.shape[:2]    

        contours, edges = find_contours(img)
        dist_append = []
        
        fractions = fractions or self.fractions

        # find hull and defects of the contour 
        for n in range(0,len(contours)):
            cnt = contours[n]
            hull = cv2.convexHull(cnt,returnPoints = False)
            defects = cv2.convexityDefects(cnt,hull)

            # initialize list to track all points on the hull as well as points with defects 
            point_track = []

            for j in range (hull.shape[0]):    
                i = hull[j]
                point = tuple(cnt[i][0][0])
                point_track.append(point)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    far = tuple(cnt[f][0])
                    point_track.append(far)

            # 4b. find max distance and plot 2 points along with longest line 
            if len(point_track) >= 2:
                [point1, point2, max_dist, point3, point4, second_dist] = find_max(point_track)
                ratio_dist = second_dist/max_dist

                # find midway points along line segment  
                midpts = get_midway( point1, point2, fractions ) 

                # slope of normal line 
                if (point2[0]-point1[0]) == 0:
                    slope = 0
                elif (point2[1]-point1[1]) == 0:
                    slope = 10000
                else:
                    slope = -1 / ( (point2[1]-point1[1]) / (point2[0]-point1[0]) )

                # 4d. loop through the middle/quarter/eighth points and plot the normal lines 
                for m, pt in enumerate(midpts):

                    x = np.linspace(0.0,cols,3)
                    y = [float(slope * i - slope * pt[0] + pt[1]) for i in x]  

                    intersect = find_intersect(x, y, cnt[:,0,0], cnt[:,0,1])

                    # distance between middle point and intersect points, normalize using length of longest line
                    dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / max_dist , 3 ) for i in intersect ]
                    split = [split_contour(i[0],i[1],pt[0],pt[1]) for i in intersect]

                    for d in range(0,len(dist)):
                        if (n+split[d],fractions[m],dist[d]) not in dist_append:
                            dist_append.append((n+split[d],fractions[m],dist[d]))

                # find midway points along second best line segment  
                if n <= 1 and ratio_dist >= .988:
                    midpts2 = get_midway( point3, point4, fractions ) 

                    # slope of normal line 
                    if (point4[0]-point3[0]) == 0:
                        slope2 = -1 / ( (point4[1]-point3[1]) / (point4[0]-point3[0] + 0.0001) )
                    elif (point4[1]-point3[1]) == 0:
                        slope2 = -1 / ( (point4[1]-point3[1] + 0.0001) / (point4[0]-point3[0]) )
                    else:
                        slope2 = -1 / ( (point4[1]-point3[1]) / (point4[0]-point3[0]) ) 

                    for m, pt in enumerate(midpts2):
                        x1 = np.linspace(0,cols,num=cols+1)
                        y1 = [slope2 * i - slope2 * pt[0] + pt[1] for i in x1] 

                        intersect = find_intersect(x1, y1, cnt[:,0,0], cnt[:,0,1])
                        if len(intersect) > 0:

                            # distance between middle point and intersect points, normalize using length of longest line
                            dist = [ round( math.sqrt( (i[0] - pt[0])**2 + (i[1] - pt[1])**2 ) / second_dist , 2 ) for i in intersect ]
                            split = [split_contour(i[0],i[1],pt[0],pt[1]) for i in intersect]
                            for d in range(0,len(dist)):
                                if (5+split[d],fractions[m],dist[d]) not in dist_append:
                                    dist_append.append((5+split[d],fractions[m],dist[d]))              

        ## create final dictionary for contour method distances    
        query = {}        
        for d in dist_append:
            query.setdefault(str((d[0],d[1])),[]).append(d[2])    

        return query         

    def compare_queries(self, query1, query2, error=0.1, fractions=None, **kwargs):
        fractions = fractions or self.fractions       
        exclude_circle = []

        # check if outer contour is circular
        for n in [0,0.5,5,5.5]:
            scores_circle = []
            for f in fractions:
                diff_circle = []
                for dist in query1.get(str((n,f)) , []):
                    f1 = 2*abs(0.5 - f)                    
                    diff_circle1 = abs(dist - np.sin(np.arccos(f1))/2)
                    if dist > 0 :
                        diff_circle.append(diff_circle1)
                diff_circle.sort()
                if len(diff_circle) >= 1:
                    partial_score_circle = max(100 - diff_circle[0]*100/error , 0)
                else:
                    partial_score_circle = 0
                scores_circle.append(partial_score_circle)

            final_score_circle = sum(scores_circle)/len(scores_circle)
            if final_score_circle >= 90:
                exclude_circle.append(n)
 
        # compare each contour between query and database image, exclude outer circle 
        contour_matrix = []
        for n in frange(0,6,0.5):
            temp_matrix = []
            if n not in exclude_circle:             
                for m in frange(0,6,0.5):
                    scores = []
                    for f in fractions:          
                        diff = []
                        for dist in query1.get(str((n,f)) , []):
                            for query_dist in query2.get(str((m,f)) , []):
                                diff1 = abs(dist - query_dist)
                                if dist > 0 and query_dist > 0:
                                    diff.append(diff1)
                        if len(diff) == 1:
                            partial_score = max(100 - diff[0]*100/error , 0)
                        elif len(diff) == 2:
                            diff.sort()
                            partial_score = max(100 - diff[0]*50/error - diff[1]*50/error, 0)
                        elif len(diff) > 2:
                            partial_score = max(100 - diff[0]*33.33/error - diff[1]*33.333/error - diff[2]*33.33/error, 0)
                        else:
                            partial_score = 0
                        scores.append(partial_score)
                        
                    temp_matrix.append( round(sum(scores)/len(scores),3))
                    
            contour_matrix.append(temp_matrix)

        all_scores = []
        for i in range(len(contour_matrix)):
            try:
                temp = max(contour_matrix[i])
            except:
                temp = 0
            all_scores.append(temp)  
            
        if len(all_scores) > 0:
            final_score = sum(all_scores)/len(all_scores)
        else:
            final_score = 0

            

        # compare each contour between query and reversed database image, exclude outer circle
        contour_matrix = []
        for n in frange(0,6,0.5):  
            temp_matrix = []
            if n not in exclude_circle:
                for m in frange(0,6,0.5):
                    scores = []
                    for f in fractions:          
                        diff = []
                        for dist in query1.get(str((n,f)) , []):
                            for query_dist in query2.get(str((m,(1-f))) , []):
                                diff1 = abs(dist - query_dist)
                                if dist > 0 and query_dist > 0:
                                    diff.append(diff1)
                        if len(diff) == 1:
                            partial_score = max(100 - diff[0]*100/error , 0)
                        elif len(diff) == 2:
                            diff.sort()
                            partial_score = max(100 - diff[0]*50/error - diff[1]*50/error, 0)
                        elif len(diff) > 2:
                            partial_score = max(100 - diff[0]*33.33/error - diff[1]*33.333/error - diff[2]*33.33/error, 0)
                        else:
                            partial_score = 0
                        scores.append(partial_score)
                        
                    temp_matrix.append( round(sum(scores)/len(scores),3))
                    
            contour_matrix.append(temp_matrix)

        all_scores = []
        for i in range(len(contour_matrix)):
            try:
                temp = max(contour_matrix[i])
            except:
                temp = 0
            all_scores.append(temp)  
            
        if len(all_scores) > 0:
            reverse_score = sum(all_scores)/len(all_scores)
        else:
            reverse_score = 0
           
        return max(final_score, reverse_score)
