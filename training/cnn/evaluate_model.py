import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import fnmatch
import csv

import statistics as stats
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(1, './utils/')
from CNN_utils import LanePoseDataset, TransCropHorizon, TransConvCoord, RandImageAugment, ToCustomTensor, showBatch

sys.path.insert(1, './testmodels/')
from CNN_Model import CNN

if sys.version_info[0] < 3:
    python_version = '2.7'
else:
    python_version = '3.x'

# Paths
path_to_home = os.environ['HOME']
path_to_models = ''.join([path_to_home, '/backups/savedModels/'])

# Read input arguments
in_args_list = list(sys.argv)
in_models_list = in_args_list[1:]

if in_models_list[0] == 'folder':
    in_files_list = [f for f in listdir(path_to_models) if isfile(join(path_to_models, f))]
    in_models_list = fnmatch.filter(in_files_list, '*Model_final')
    print('evaluating all files in dir', path_to_models)


# CUDA for PyTorch
# parser = argparse.ArgumentParser(description='PyTorch Example')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# args = parser.parse_args()
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Transformation parameters for later use
image_res = 64
# pixel_shift_h = 2
# use_pixelshift = True
as_grayscale = True
use_convcoord = True

batch_size = 1

transforms = transforms.Compose([
    transforms.Resize(image_res),
    TransCropHorizon(0.5, set_black=False),
    # transforms.RandomCrop(, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
    RandImageAugment(augment_white_balance = False,
                    augment_brightness = False,
                    augment_contrast = False),
    transforms.Grayscale(num_output_channels=1),
    TransConvCoord(),
    ToCustomTensor(use_convcoord),
    # transforms.Normalize(mean = [0.3,0.5,0.5],std = [0.21,0.5,0.5])
    ])


validation_set = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r5/']), csvFilename='output_oli_eval.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r5/images/']), transform = transforms)
validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle=True)

# Model class must be defined somewhere
for in_model in in_models_list:
    try:
        model = torch.load(''.join([path_to_models, in_model]), map_location=device)
        print('*******************************************************************')
        print(in_model)

        model.eval()
        model.double().to(device=device)

        model_eval_error_d_list = []
        model_eval_error_theta_list = []

        correct_d = 0.0
        correct_theta = 0.0
        abs_tol_d = 0.03
        abs_tol_theta = 3.1415/180*8
        total = 0

        with torch.set_grad_enabled(False):
            for i, (images, poses) in enumerate(validation_loader):
                # images = images.unsqueeze(1)
                images = images.double().to(device=device)
                outputs = model(images)

                # Recover actual theta
                # print(outputs[:,1])
                outputs[:,1] = outputs[:,1]*3.1415
                # print(outputs[:,1])
                # showBatch(images, poses, as_grayscale=as_grayscale, batch_size=batch_size, outputs=outputs)
                # print(outputs)
                # print('output_d: {:.3f}, gt_d: {:.3f}, output_theta: {:.3f}, gt_theta: {:.3f}'
                #     .format(outputs[0][0], poses[0][0], outputs[0][1], poses[0][1]))
                # print('error_d: {:.3f}, error_theta: {:.3f}'
                #     .format(outputs[0][0]-poses[0][0],outputs[0][1]-poses[0][1]))

                error_d = outputs[0,0].item()-poses[0,0].item()
                error_theta = outputs[0,1].item()-poses[0,1].item()

                model_eval_error_d_list.append(error_d)
                model_eval_error_theta_list.append(error_theta)

                total += 1
                if abs(poses[0][0]-outputs[0][0]) < abs_tol_d:
                    correct_d += 1
                if abs(poses[0][1]-outputs[0][1]) < abs_tol_theta:
                    correct_theta += 1

        stdev_d = stats.stdev(model_eval_error_d_list)
        stdev_theta = stats.stdev(model_eval_error_theta_list)

        print('stdev d: {:.5f}, stdev theta: {:.5f}'
             .format(stdev_d, stdev_theta))

        print('nr. of samples: {:.0f}'
             .format(total))
        print('abs_tol_d [m]: {:.2f}, accuracy_d: {:.1f}%'
             .format(abs_tol_d, (correct_d/total*100)))
        print('abs_tol_theta [grad]: {:.2f}, accuracy_theta: {:.1f}%'
             .format(abs_tol_theta/3.1415*180, (correct_theta/total*100)))

        # write csv files
        if python_version == '3.x':
            w_var = 'w'
        else:
            w_var = 'wb'

        with open(''.join([path_to_models, in_model,'_eval','.csv']), w_var) as csv_eval:
            wr = csv.writer(csv_eval, quoting=csv.QUOTE_ALL)
            wr.writerow(['picture', 'eval_error_d', 'eval_error_theta', stdev_d, stdev_theta])
            for i in range(0, len(model_eval_error_d_list)):
                wr.writerow(['picturename', model_eval_error_d_list[i], model_eval_error_theta_list[i]])
    except Exception as e: print(e)
