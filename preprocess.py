import numpy as np
from PIL import Image
import pdb
import os
import cv2
from scipy.io import loadmat,savemat
from collections import defaultdict


# args
import argparse
parser = argparse.ArgumentParser(description='args for preprocessing Market-Sketch-1K')
parser.add_argument('--data_path', type=str, default='/data3/lkj/rebuttal_sketchreid/dataset/market-mix-cross', help='path to dataset, and where you store processed attributes')
parser.add_argument('--attribute_path', type=str, default='/data3/lkj/rebuttal_sketchreid/dataset/market-mix-cross', help='path to Market1501\'s attribute')
parser.add_argument('--image_width', type=int, default=144, help='image width')
parser.add_argument('--image_height', type=int, default=288, help='image height')
parser.add_argument('--train_style', type=str, default='A', help='styles: any combination of A-F. For example: B, EF, ACEF...')
parser.add_argument('--train_mq', action='store_true', help='train with multi-query')
args = parser.parse_args()

data_path = args.data_path
attribute_path = args.attribute_path
fix_image_width = args.image_width
fix_image_height = args.image_height

# load files
files_rgb = os.listdir(data_path+'/photo/train')
files_sk = {s: os.listdir(f'{data_path}/sketch/{s}/train') for s in args.train_style}

# relabel
pid_container = set()
for s in files_sk.keys():
    files = files_sk[s]
    for img_path in files:
        pid = int(img_path[:4])
        pid_container.add(pid)
pid2label = {pid:label for label, pid in enumerate(pid_container)}

# read photos
def read_imgs_single(train_image, dir):
    train_img = []
    train_label = []
    
    for img_path in train_image:
        # img
        if not int(img_path[:4]) in pid2label.keys() or not img_path[-4:] == '.jpg':
            continue

        img = Image.open(dir+'/'+img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)
        if len(pix_array.shape) == 2:
            pix_array = cv2.cvtColor(pix_array,cv2.COLOR_GRAY2RGB)
        train_img.append(pix_array) 
        
        # label
        pid = int(img_path[:4])

        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label).astype('int')

def read_sketches_single(train_image, dir):
    train_img = []
    train_label = []

    _train_image = defaultdict(list)
    for s in files_sk.keys():
        train_image = files_sk[s]
        for img_path in train_image:
            pid = int(img_path[:4])
            _train_image[pid].append(f'sketch/{s}/train/{img_path}')

    for pid in sorted(_train_image.keys()):
            # add image: [6, *img.shape] and style:[1]
            img_paths = _train_image[pid]
            if not pid in pid2label.keys():
                continue
            pid = pid2label[pid]

            for img_path in img_paths:
                if not img_path[-4:] == '.jpg':
                    continue
                img = Image.open(dir+'/'+img_path)
                img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
                pix_array = np.array(img)
                if len(pix_array.shape) == 2:
                    pix_array = cv2.cvtColor(pix_array,cv2.COLOR_GRAY2RGB)


                train_img.append(pix_array)
                # add label
                train_label.append(pid)
    return train_img, train_label

def read_sketch_multi(train_image, dir):
    train_img = []
    train_label = []
    styles = []

    _train_image = defaultdict(list)
    for s in files_sk.keys():
        train_image = files_sk[s]
        for img_path in train_image:
            pid = int(img_path[:4])
            _train_image[pid].append(f'sketch/{s}/train/{img_path}')

    for pid in sorted(_train_image.keys()):
        # add image: [6, *img.shape] and style:[1]
        if not pid in pid2label.keys():
            continue
        img_paths = _train_image[pid]
        style = len(img_paths)
        styles.append(style)
        imgs = []
        for img_path in img_paths:
            if not img_path[-4:] == '.jpg':
                continue
            img = Image.open(dir+'/'+img_path)
            img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array = cv2.cvtColor(pix_array,cv2.COLOR_GRAY2RGB)
            imgs.append(pix_array)
        for _ in range(6-style):
            padImg = np.zeros(imgs[0].shape).astype(pix_array.dtype)
            imgs.append(padImg)

        imgs = np.array(imgs)
        train_img.append(imgs)
        # add label
        pid = pid2label[pid]
        train_label.append(pid)
    
    return np.array(train_img), np.array(train_label).astype('int'), np.array(styles).astype('int')

def read_attributes():
    tmp = [[],[]]
    names = ['gender', 'hair', 'up', 'down', 'clothes', 'hat', 'backpack', 'bag', 'handbag', 'age',\
        'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen',\
        'downblack', 'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue', 'downgreen', 'downbrown']

    # save all attribute -> (1501, 27)
    mat = loadmat(f'{attribute_path}/market_attribute.mat')['market_attribute']
    newM = np.zeros((27,1502))
    for i in range(751):
        m = mat[0][0][1][0][0]
        for j in range(27):
            newM[j][int(m[27][0][i])] = m[names[j]][0][i]
        tmp[0].append(int(m[27][0][i]))

    for i in range(750):
        m = mat[0][0][0][0][0]
        for j in range(27):
            newM[j][int(m[27][0][i])] = m[names[j]][0][i]
        tmp[1].append(int(m[27][0][i]))

    # save train attribute and relabel
    trainM = np.zeros((len(pid2label),27))
    for id,l in pid2label.items():
        trainM[l] = newM.T[id]

    return trainM

if __name__=='__main__':
    os.makedirs(f'{data_path}/feature', exist_ok=True)
    # rgb imges
    train_photo, train_label = read_imgs_single(files_rgb, f'{data_path}/photo/train')
    np.save(f'{data_path}/feature/train_rgb_img.npy', train_photo)
    np.save(f'{data_path}/feature/train_rgb_label.npy', train_label)

    # sketches
    if len(args.train_style)==1 or not args.train_mq:
        train_sketch, train_label = read_sketches_single(files_sk, data_path)
        np.save(f'{data_path}/feature/train_sk_img_{args.train_style}.npy', train_sketch)
        np.save(f'{data_path}/feature/train_sk_label_{args.train_style}.npy', train_label)
    elif len(args.train_style)>1:
        train_sketch, train_label, train_style = read_sketch_multi(files_sk, f'{data_path}')
        np.save(f'{data_path}/feature/train_sk_img_{args.train_style}.npy', train_sketch)
        np.save(f'{data_path}/feature/train_sk_label_{args.train_style}.npy', train_label)
        np.save(f'{data_path}/feature/train_sk_numStyle_{args.train_style}.npy', train_style)
    
    # attributes
    attributes = read_attributes()
    savemat(f'{data_path}/market_attribute_train.mat', {'data':attributes.astype(int)})