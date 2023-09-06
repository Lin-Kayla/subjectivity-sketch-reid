from inspect import Attribute
import numpy as np
from PIL import Image
import torch.utils.data as data
import cv2
import os
import torch
from utils import get_textInput
from scipy.io import loadmat
import clip

class Mask1kData_single(data.Dataset):
    def __init__(self, data_dir, train_style, transform=None, colorIndex = None, thermalIndex = None):
        
        self.attribute = loadmat(os.path.join(data_dir, 'market_attribute_train.mat'))['data']

        # Load training images (path) and labels
        train_color_image = np.load(os.path.join(data_dir, 'feature', 'train_rgb_img.npy'))
        self.train_color_label = np.load(os.path.join(data_dir, 'feature', 'train_rgb_label.npy'))

        train_sketch_image = np.load(os.path.join(data_dir, 'feature', 'train_sk_img_' + train_style + '.npy'))
        self.train_sketch_label = np.load(os.path.join(data_dir, 'feature', 'train_sk_label_' + train_style + '.npy'))
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_sketch_image = train_sketch_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_sketch_image[self.tIndex[index]], self.train_sketch_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        text1 = get_textInput(self.attribute[target1])
        text1 = clip.tokenize(text1).detach().int().squeeze(0)
        text2 = get_textInput(self.attribute[target2])
        text2 = clip.tokenize(text2).detach().int().squeeze(0)

        return img1, img2, text1, text2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class Mask1kData_multi(data.Dataset):
    def __init__(self, data_dir, train_style, transform=None, colorIndex = None, thermalIndex = None):
        
        self.attribute = loadmat(os.path.join(data_dir, 'market_attribute_train.mat'))['data']

        # Load training images (path) and labels
        train_color_image = np.load(os.path.join(data_dir, 'feature', 'train_rgb_img.npy'))
        self.train_color_label = np.load(os.path.join(data_dir, 'feature', 'train_rgb_label.npy'))

        train_sketch_image = np.load(os.path.join(data_dir, 'feature', 'train_sk_img_'+ train_style +'.npy'))
        self.train_sketch_label = np.load(os.path.join(data_dir, 'feature', 'train_sk_label_'+ train_style +'.npy'))

        self.train_sketch_style = np.load(os.path.join(data_dir, 'feature', 'train_sk_numStyle_'+ train_style +'.npy'))
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_sketch_image = train_sketch_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_sketch_image[self.tIndex[index]], self.train_sketch_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = torch.stack([self.transform(img) for img in img2],0)

        text1 = get_textInput(self.attribute[target1])
        text1 = clip.tokenize(text1).detach().int().squeeze(0)
        text2 = get_textInput(self.attribute[target2])
        text2 = clip.tokenize(text2).detach().int().squeeze(0)

        style = self.train_sketch_style[self.tIndex[index]]

        return img1, img2, text1, text2, target1, target2, style

    def __len__(self):
        return len(self.train_color_label)
  
class PKUDataAttr(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(os.path.join(data_dir, 'train_rgb_resized_img.npy'))
        self.train_color_label = np.load(os.path.join(data_dir, 'train_rgb_resized_label.npy'))

        train_sketch_image = np.load(os.path.join(data_dir, 'train_ir_resized_img.npy'))
        self.train_sketch_label = np.load(os.path.join(data_dir, 'train_ir_resized_label.npy'))
        
        self.attribute= loadmat(os.path.join(data_dir, 'PKU_attribute_train.mat'))['data']
        
        self.train_color_image   = train_color_image
        self.train_sketch_image = train_sketch_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_sketch_image[self.tIndex[index]], self.train_sketch_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)


        # m = self.attribute[target2]
        text1 = get_textInput(self.attribute[target1])
        text1 = clip.tokenize(text1).detach().int().squeeze(0)
        text2 = get_textInput(self.attribute[target2])
        text2 = clip.tokenize(text2).detach().int().squeeze(0)

        return img1, img2, text1, text2, target1, target2 

    def __len__(self):
        return len(self.train_color_label)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array = cv2.cvtColor(pix_array, cv2.COLOR_GRAY2RGB)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        
        print('gall size',test_image.shape)
        print('gall label size',len(test_label))

        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class TestData_ensemble(data.Dataset):
    def __init__(self, test_img_file, test_label, test_style, transform=None, img_size = (144,288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array = cv2.cvtColor(pix_array, cv2.COLOR_GRAY2RGB)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        
        print('gall size',test_image.shape)
        print('gall label size',len(test_label))

        self.test_image = test_image
        self.test_label = test_label
        self.test_style = test_style
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        style = self.test_style[index]
        return img1, target1,style

    def __len__(self):
        return len(self.test_image)

class TestData_multi(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):
        test_image = []
        test_style = []
        for files in test_img_file:
            _test_image =  []
            test_style.append(len(files))
            for f in files:
                img = Image.open(f)
                img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
                pix_array = np.array(img)
                if len(pix_array.shape) == 2:
                    pix_array = cv2.cvtColor(pix_array, cv2.COLOR_GRAY2RGB)
                _test_image.append(pix_array)
            for _ in range(6-len(files)):
                _test_image.append(np.zeros_like(pix_array))
            test_image.append(np.array(_test_image))
        test_image = np.array(test_image)
        
        print('query size',test_image.shape)
        print('query label size',len(test_label))

        self.test_style = test_style
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1, style1 = self.test_image[index],  self.test_label[index], self.test_style[index]
        img1 = torch.stack([self.transform(img) for img in img1],0)
        return img1, target1, style1

    def __len__(self):
        return len(self.test_image)
