import os
import random
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import errno
import math
from PIL import Image
import cv2

from torchvision import transforms

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)

def get_textInput(idx):
    # idx: (27)
    
    gender = ['person', 'man', 'woman'][idx[0]]
    annun = ['He or she', 'He', 'She'][idx[0]]
    hairLen = ['', 'short', 'long'][idx[1]]
    sleeveLen = ['', 'long', 'short'][idx[2]]
    lowerLen = ['', 'long', 'short'][idx[3]]
    lowerType = ['clothes', 'dress', 'pants'][idx[4]]
    accesories = ' with ' + ' and '.join(np.array(['hat', 'backpack', 'bag', 'handbag'])[np.where(idx[5:9]==2)]) + '.' if 2 in idx[5:9] else '.'
    age = ['', 'young', 'teenage', 'adult', 'old'][idx[9]]
    person = age + ' ' + gender
    upColorAll = np.array(['black', 'white', 'red', 'purple', 'yellow', 'gray', 'blue', 'green'])
    upColor = ' and '.join(upColorAll[np.where(idx[10:18]==2)]) + ' ' if 2 in idx[10:18] else ''
    downColorAll = np.array(['black', 'white', 'pink', 'purple', 'yellow', 'gray', 'blue', 'green', 'brown'])
    downColor = ' and '.join(downColorAll[np.where(idx[18:]==2)]) + ' ' if 2 in idx[18:] else ''
    answer = ' '.join(['An' if idx[9] in [3,4] else 'A', person, 'with', hairLen, 'hair.', annun, 'is in', upColor, sleeveLen,'sleeve clothes, and wears',\
         downColor, lowerLen, lowerType]) + accesories

    # print(answer)
    return answer.replace('  ',' ').replace('  ',' ').replace('  ',' ')


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data_1(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    # y_a, y_b = y, y[index]
    return mixed_x #, y_a, y_b, lam

def mixup_data_2(x1,x2, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x1.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index,:]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index,:]
    # y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2 #, y_a, y_b, lam

def mixup_criterion_ori(y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion(criterion, output, y_a, y_b, lam, type='id'):
    if type=='id':
        return lam*criterion(output, y_a) + (1-lam)*criterion(output, y_b)
    elif type=='tri':
        loss_tri_a, batch_acc_a = criterion(output, y_a)
        loss_tri_b, batch_acc_b = criterion(output, y_b)
        return lam*loss_tri_a+(1-lam)*loss_tri_b, lam*batch_acc_a+(1-lam)*batch_acc_b

class RandomErasing(object):
    def __init__(self, EPSILON = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = [255,255,255]
       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.EPSILON:
        #     return img

        img = transforms.PILToTensor()(img)
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                img = transforms.ToPILImage()(img)
                return img
        img = transforms.ToPILImage()(img)
        return img

class Erosion(object):
    def __init__(self, img_h=288,img_w=144, EPSILON = 0.5):
        self.kernel = [[1,1,1],[1,1,1],[1,1,1]]
        self.kernel_h,self.kernel_w = np.shape(self.kernel)
        self.kernel_cx = self.kernel_w // 2
        self.kernel_cy = self.kernel_h // 2
        self.img_h = img_h
        self.img_w = img_w
        self.EPSILON = EPSILON

    def __call__(self, img):
        # if random.uniform(0, 1) > self.EPSILON:
        #     return img
        img = transforms.transforms.PILToTensor()(img)
        final_image_erosion = np.zeros(img.shape)
        for i in range(self.img_h):
            for j in range(self.img_w):
                minPic = np.min(img[0, max(0,i-1):min(self.img_h,i+2), max(0,j-1):min(self.img_w,j+2)].numpy())
                final_image_erosion[0][i][j] = minPic
        final_image_erosion = np.ones(final_image_erosion.shape)*256-final_image_erosion
        return transforms.ToPILImage()(torch.tensor(final_image_erosion))

class Dilation(object):
    def __init__(self, img_h=288,img_w=144, EPSILON = 0.5):
        self.kernel = [[1,1,1],[1,1,1],[1,1,1]]
        self.kernel_h,self.kernel_w = np.shape(self.kernel)
        self.kernel_cx = self.kernel_w // 2
        self.kernel_cy = self.kernel_h // 2
        self.img_h = img_h
        self.img_w = img_w
        self.EPSILON = EPSILON

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img

        img = transforms.transforms.PILToTensor()(img)
        final_image_dilation = np.zeros(img.shape)
        for i in range(self.img_h):
            for j in range(self.img_w):
                max = 0
                for x in range(i-self.kernel_cx,i+self.kernel_cx+1):
                    for y in range(j-self.kernel_cy,j+self.kernel_cy+1):
                        if(x>=0 and x<self.img_h and y>=0 and y<self.img_w):
                                if(img[0][x][y]>max):
                                    max = img[0][x][y]
                final_image_dilation[0][i][j] = max
        final_image_dilation = np.ones(final_image_dilation.shape)*256-final_image_dilation
        return transforms.ToPILImage()(torch.tensor(final_image_dilation))

class Flip(object):
    """
    This class is used to mirror images through the x or y axes.
    The class allows an image to be mirrored along either
    its x axis or its y axis, or randomly.
    """
    def __init__(self, probability, top_bottom_left_right):
        """
        The direction of the flip, or whether it should be randomised, is
        controlled using the :attr:`top_bottom_left_right` parameter.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param top_bottom_left_right: Controls the direction the image should
         be mirrored. Must be one of ``LEFT_RIGHT``, ``TOP_BOTTOM``, or
         ``RANDOM``.
         - ``LEFT_RIGHT`` defines that the image is mirrored along its x axis.
         - ``TOP_BOTTOM`` defines that the image is mirrored along its y axis.
         - ``RANDOM`` defines that the image is mirrored randomly along
           either the x or y axis.
        """
        self.probability = probability
        self.top_bottom_left_right = top_bottom_left_right

    def __call__(self, image):
        """
        Mirror the image according to the `attr`:top_bottom_left_right`
        argument passed to the constructor and return the mirrored image.
        :param images: The image(s) to mirror.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        random_axis = random.randint(0, 1)

        def do(image):
            if self.top_bottom_left_right == "LEFT_RIGHT":
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.top_bottom_left_right == "TOP_BOTTOM":
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            elif self.top_bottom_left_right == "RANDOM":
                if random_axis == 0:
                    return image.transpose(Image.FLIP_LEFT_RIGHT)
                elif random_axis == 1:
                    return image.transpose(Image.FLIP_TOP_BOTTOM)

        augmented_images = do(image)

        return augmented_images

class GaussianDistortion(object):
    """
    This class performs randomised, elastic gaussian distortions on images.
    """
    def __init__(self, probability, grid_width, grid_height, magnitude, corner, method, mex, mey, sdx, sdy):
        """
        As well as the probability, the granularity of the distortions
        produced by this class can be controlled using the width and
        height of the overlaying distortion grid. The larger the height
        and width of the grid, the smaller the distortions. This means
        that larger grid sizes can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :param corner: which corner of picture to distort.
         Possible values: "bell"(circular surface applied), "ul"(upper left),
         "ur"(upper right), "dl"(down left), "dr"(down right).
        :param method: possible values: "in"(apply max magnitude to the chosen
         corner), "out"(inverse of method in).
        :param mex: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param mey: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdx: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdy: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :type corner: String
        :type method: String
        :type mex: Float
        :type mey: Float
        :type sdx: Float
        :type sdy: Float
        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:
        .. math::
         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        self.randomise_magnitude = True
        self.corner = corner
        self.method = method
        self.mex = mex
        self.mey = mey
        self.sdx = sdx
        self.sdy = sdy

    def __call__(self, image):
        """
        Distorts the passed image(s) according to the parameters supplied
        during instantiation, returning the newly distorted image.
        :param images: The image(s) to be distorted.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        w, h = image.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(math.floor(w / float(horizontal_tiles)))
        height_of_square = int(math.floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        def sigmoidf(x, y, sdx=0.05, sdy=0.05, mex=0.5, mey=0.5, const=1):
            sigmoid = lambda x1, y1:  (const * (math.exp(-(((x1-mex)**2)/sdx + ((y1-mey)**2)/sdy) )) + max(0,-const) - max(0, const))
            xl = np.linspace(0,1)
            yl = np.linspace(0, 1)
            X, Y = np.meshgrid(xl, yl)

            Z = np.vectorize(sigmoid)(X, Y)
            mino = np.amin(Z)
            maxo = np.amax(Z)
            res = sigmoid(x, y)
            res = max(((((res - mino) * (1 - 0)) / (maxo - mino)) + 0), 0.01)*self.magnitude
            return res

        def corner(x, y, corner="ul", method="out", sdx=0.05, sdy=0.05, mex=0.5, mey=0.5):
            ll = {'dr': (0, 0.5, 0, 0.5), 'dl': (0.5, 1, 0, 0.5), 'ur': (0, 0.5, 0.5, 1), 'ul': (0.5, 1, 0.5, 1), 'bell': (0, 1, 0, 1)}
            new_c = ll[corner]
            new_x = (((x - 0) * (new_c[1] - new_c[0])) / (1 - 0)) + new_c[0]
            new_y = (((y - 0) * (new_c[3] - new_c[2])) / (1 - 0)) + new_c[2]
            if method == "in":
                const = 1
            else:
                if method == "out":
                    const =- 1
                else:
                    const = 1
            res = sigmoidf(x=new_x, y=new_y,sdx=sdx, sdy=sdy, mex=mex, mey=mey, const=const)

            return res

        def do(image):

            for a, b, c, d in polygon_indices:
                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]

                sigmax = corner(x=x3/w, y=y3/h, corner=self.corner, method=self.method, sdx=self.sdx, sdy=self.sdy, mex=self.mex, mey=self.mey)
                dx = np.random.normal(0, sigmax, 1)[0]
                dy = np.random.normal(0, sigmax, 1)[0]
                polygons[a] = [x1, y1,
                               x2, y2,
                               x3 + dx, y3 + dy,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
                polygons[b] = [x1, y1,
                               x2 + dx, y2 + dy,
                               x3, y3,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
                polygons[c] = [x1, y1,
                               x2, y2,
                               x3, y3,
                               x4 + dx, y4 + dy]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
                polygons[d] = [x1 + dx, y1 + dy,
                               x2, y2,
                               x3, y3,
                               x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = do(image)

        return augmented_images

class ZoomRandom(object):
    """
    This class is used to zoom into random areas of the image.
    """

    def __init__(self, probability, percentage_area, randomise):
        """
        Zooms into a random area of the image, rather than the centre of
        the image, as is done by :class:`Zoom`. The zoom factor is fixed
        unless :attr:`randomise` is set to ``True``.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: A value between 0.1 and 1 that represents the
         area that will be cropped, with 1 meaning the entire area of the
         image will be cropped and 0.1 mean 10% of the area of the image
         will be cropped, before zooming.
        :param randomise: If ``True``, uses the :attr:`percentage_area` as an
         upper bound, and randomises the zoom level from between 0.1 and
         :attr:`percentage_area`.
        """
        self.probability = probability
        self.percentage_area = percentage_area
        self.randomise = randomise

    def __call__(self, image):
        """
        Randomly zoom into the passed :attr:`images` by first cropping the image
        based on the :attr:`percentage_area` argument, and then resizing the
        image to match the size of the input area.
        Effectively, you are zooming in on random areas of the image.
        :param images: The image to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        if self.randomise:
            r_percentage_area = round(random.uniform(0.1, self.percentage_area), 2)
        else:
            r_percentage_area = self.percentage_area

        w, h = image.size
        w_new = int(math.floor(w * r_percentage_area))
        h_new = int(math.floor(h * r_percentage_area))

        random_left_shift = random.randint(0, (w - w_new))  # Note: randint() is from uniform distribution.
        random_down_shift = random.randint(0, (h - h_new))

        def do(image):
            image = image.crop((random_left_shift, random_down_shift, w_new + random_left_shift, h_new + random_down_shift))

            return image.resize((w, h), resample=Image.BICUBIC)

        augmented_images = do(image)

        return augmented_images

class erosion(object):
    def __init__(self):
        self.kernel = np.ones((3,3))
        self.iteration=1
        self.EPSILON = 0.5

    def __call__(self, image):
        # if random.uniform(0, 1) > self.EPSILON:
        #     return image
        # print(self.kernel)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        img = cv2.erode(img,kernel=self.kernel,iterations=self.iteration)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return image

class dilation(object):
    def __init__(self):
        self.kernel = np.ones((3,3))
        self.iteration=1
        self.EPSILON = 0.5

    def __call__(self, image):
        if random.uniform(0, 1) > self.EPSILON:
            return image
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        img = cv2.dilate(img,kernel=self.kernel,iterations=self.iteration)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return image
