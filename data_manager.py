from __future__ import print_function, absolute_import
import os
import numpy as np
import random

def process_test_market(img_dir, modal = 'photo'):
    if modal=='photo':
        input_data_path = img_dir + 'photo/query' 
    elif modal=='sketch':
        input_data_path = img_dir + 'sketch/query'
    
    data_list = os.listdir(input_data_path)
    file_image = [input_data_path + '/' + path for path in data_list]
    file_label = [path[:4] for path in data_list]

    return file_image, np.array(file_label)

def process_test_pku(img_dir, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'photo/query' 
    elif modal=='thermal':
        input_data_path = img_dir + 'sketch/query'
    
    data_list = os.listdir(input_data_path)
    file_image = [input_data_path + '/' + path for path in data_list]

    if modal=='visible':
        file_label = [path.split('_')[0] for path in data_list]
    else:
        file_label = [path.split('.')[0] for path in data_list]
    return file_image, np.array(file_label)

def process_test_mask1k_single(img_dir, test_style):
    file_image = []
    file_label = []
    
    for s in test_style:

        input_data_path = img_dir + 'sketch/' + s + '/query'
    
        data_list = os.listdir(input_data_path)
        file_image.extend([input_data_path + '/' + path for path in data_list])
        file_label.extend([path[:4] for path in data_list])

    return file_image, np.array(file_label)

def process_test_market_ensemble(img_dir, test_style):
    file_image = []
    file_label = []
    file_style = []
    
    for s in test_style:

        input_data_path = img_dir + 'sketch/' + s + '/query'
    
        data_list = os.listdir(input_data_path)
        file_image.extend([input_data_path + '/' + path for path in data_list])
        file_label.extend([path[:4] for path in data_list])
        file_style.extend([s for _ in data_list])

    return file_image, np.array(file_label), np.array(file_style)

def process_test_mask1k_multi(img_dir, test_style):

    file_label = []
    file_image_dict = {}
    for s in test_style:
        input_data_path = img_dir + 'sketch/' + s + '/query'
        data_list = os.listdir(input_data_path)
        
        for path in data_list:
            if path[:4] not in file_label:
                file_label.append(path[:4])
            if path[:4] not in file_image_dict.keys():
                file_image_dict[path[:4]] = []
            file_image_dict[path[:4]].append(input_data_path + '/' + path)

    file_image = [file_image_dict[label] for label in file_label]

    return file_image, np.array(file_label)

    input_data_path = img_dir + 'sketch/query'

    data_list = os.listdir(input_data_path)

    file_label = []
    file_image_dict = {}
    file_style={}
    for path in data_list:
        if path[:4] not in file_label:
            file_label.append(path[:4])
        if path[:4] not in file_image_dict.keys():
            file_image_dict[path[:4]] = []
            file_style[path[:4]]=[0,0,0,0,0,0]
        file_style[path[:4]][int(path[6]) - 1]=1
        file_image_dict[path[:4]].append(input_data_path + '/' + path)
    
    file_style_out = [file_style[style] for style in file_style] 
    file_image = [file_image_dict[label] for label in file_label]

    return file_image, np.array(file_label), np.array(file_style_out)