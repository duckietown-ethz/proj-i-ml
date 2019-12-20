# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import sys
import argparse
import time
import csv
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import statistics as stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(1, './utils/')
from CNN_utils import LanePoseDataset, TransCropHorizon, TransConvCoord, RandImageAugment, ToCustomTensor, TransRandomErasing, showBatch

sys.path.insert(1, './model/')
from CNN_Model import CNN

# different csv wirte arguments depending on python version
if sys.version_info[0] < 3:
    python_version = '2.7'
else:
    python_version = '3.x'

# path to home
path_to_home = os.environ['HOME']



time_start = time.time()


# Read input arguments
in_arg_list = list(sys.argv)
in_program_name = in_arg_list[0]    # train_model.py
in_lr = float(in_arg_list[1])       # lr = [0.0001, 0.1]
in_bs = int(in_arg_list[2])         # bs = 16
in_epochs = int(in_arg_list[3])     # epoch = [20, 200]
in_gpu = in_arg_list[4]             # gpu = 'cuda:n' where n = [0, 1, 2, 3, 4, 5, 6 ,7]
in_workers = int(in_arg_list[5])    # cpu workers [1, 8]

if len(in_arg_list)>6:
    in_note = in_arg_list[6]   # cpu workers [1, 8
    in_filename_note = "".join(["_", in_note])
else:
    in_filename_note = ""

# CUDA for PyTorch
use_DataParallel = False
# parser = argparse.ArgumentParser(description='PyTorch Example')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# args = parser.parse_args()
# device = None
if torch.cuda.is_available():
    device = torch.device(in_gpu)
else:
    device = torch.device('cpu')

# Transformation parameters for later use
image_res = 64
# pixel_shift_h = 2
# use_pixelshift = True
as_grayscale = True
use_convcoord = False

transforms = transforms.Compose([
    transforms.Resize(image_res),
    TransCropHorizon(0.5, set_black=False),
    # transforms.RandomCrop(, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
    RandImageAugment(augment_white_balance = False,
                    augment_brightness = True, brightness_augment_max_deviation = 0.6, brightness_augment_sigma = 0.2,
                    augment_contrast = True, contrast_augment_max_deviation = 0.6, contrast_augment_sigma = 0.2),
    transforms.Grayscale(num_output_channels=1),
    # TransConvCoord(),
    ToCustomTensor(use_convcoord),
    transforms.RandomErasing(p=0.4, scale=(0.05, 0.1), ratio=(0.3, 4), value=0.2, inplace=False),
    # transforms.Normalize(mean = [0.3,0.5,0.5],std = [0.21,0.5,0.5])
    ])

# Hyperparameters
num_epochs = in_epochs
batch_size = in_bs
learning_rate = in_lr



#################################
#     Load training Datasets    #
#################################

# Trainings Sets
training_set_r1 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r1/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r1/images/']), transform = transforms)
training_set_r2 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r2/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r2/images/']), transform = transforms)
training_set_r3 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r3/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r3/images/']), transform = transforms)
training_set_r4 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r4/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r4/images/']), transform = transforms)
training_set_r5 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r5/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r5/images/']), transform = transforms)
training_set_r6 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot04_r6/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot04_r6/images/']), transform = transforms)
training_set_r7 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot05_r7/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot05_r7/images/']), transform = transforms)
training_set_r8 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot05_r8/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot05_r8/images/']), transform = transforms)
training_set_r9 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot05_r9/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot05_r9/images/']), transform = transforms)
training_set_r10 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot05_r10/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot05_r10/images/']), transform = transforms)
training_set_r11 = LanePoseDataset(csvPath=''.join([path_to_home,'/data_LanePose/autobot05_r11/']), csvFilename='output_pose.csv', imgPath=''.join([path_to_home,'/data_LanePose/autobot05_r11/images/']), transform = transforms)

training_set_list = ['training_set_r1',
                    'training_set_r2',
                    'training_set_r3',
                    'training_set_r4',
                    'training_set_r5',
                    'training_set_r6',
                    'training_set_r7',
                    'training_set_r8',
                    'training_set_r9',
                    'training_set_r10',
                    'training_set_r11',]

training_set_dict = {}
for training_set in training_set_list:
    training_set_dict.update({str(training_set): eval(training_set)})

training_set = torch.utils.data.ConcatDataset(training_set_dict.values())
training_loader = DataLoader(training_set, batch_size = batch_size, shuffle=True, num_workers=in_workers)



model = OurCNN(as_gray=as_grayscale,use_convcoord=use_convcoord)

if use_DataParallel:
    if torch.cuda.device_count() > 1:
        device_ids=[0, 1, 2, 3]
        print("Model uses", len(device_ids),'of', torch.cuda.device_count() , "GPUs!")
        model = nn.DataParallel(model, device_ids = device_ids)

# model.double().to(device=device)
model.double().to(device=device)
print(model)

# Loss and optimizer
# criterion = nn.MSELoss(reduction='sum')
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)

# Train the model
total_step = len(training_loader)
model_save_name = ''.join(['./savedModels/CNN_',str(time_start),'_lr',str(learning_rate),'_bs',str(batch_size),'_epo',str(num_epochs)])

model_loss_list = []
model_batch_list = []
model_epoch_list = []
model_time_epoch_list = []
for epoch in range(num_epochs):
    time_epoch_start = time.time()

    epoch_loss_list = []
    epoch_batch_list = []
    epoch_epoch_list = []
    loss = None
    outputs = None
    for i, (images, poses) in enumerate(training_loader):

        # Normalize pose theta [-1, 1]
        poses[:,1] = poses[:,1]/3.1415

        # Assign Tensors to Cuda device
        images = images.double().to(device=device)
        poses = poses.to(device=device)
        # print('Images Tensor is on device:',images.device)
        # print('Poses Tensor is on device:',poses.device)
        # Feed model

        outputs = model(images)

        # print(images)
        showBatch(images, poses, as_grayscale=as_grayscale, batch_size=batch_size)
        # loss_old = criterion(poses, outputs)
        # print(loss_old)

        # Loss calculations
        loss_weights = [1, 1]
        loss_d = criterion(poses[:,0], outputs[:,0])
        loss_theta = criterion(poses[:,1], outputs[:,1])
        loss = ((loss_d*loss_weights[0])) #+ loss_theta*loss_weights[1])/2)

        epoch_loss_list.append(loss.item())
        # epoch_batch_list.append(i)
        epoch_epoch_list.append(epoch)

        # Backprop and perform optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % batch_size == 0:
            print('Epoch [{}/{}], Step [{}/{}], Item [{}/{}], Loss: {:.6f}' #, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, batch_size*(i+1), len(training_set), epoch_loss_list[-1]))

            # print('output and pose')
            # print(outputs)
            # print(poses)

	# Delete variables
        del loss
        del outputs

    time_epoch_end = time.time()
    time_epoch = time_epoch_end-time_epoch_start

    epoch_mean_loss = stats.mean(epoch_loss_list)
    model_loss_list.append(epoch_mean_loss)
    model_epoch_list.append(epoch)
    model_time_epoch_list.append(time_epoch)

    if (epoch + 1) % 10 == 0:
        torch.save(model, ''.join([model_save_name,'_Model_temp', in_filename_note]))

torch.save(model, ''.join([model_save_name,'_Model_final', in_filename_note]))


# write csv files
if python_version == '3.x':
    w_var = 'w'
else:
    w_var = 'wb'

with open(''.join([model_save_name,'_config','.csv']), w_var) as csv_config:
    wr = csv.writer(csv_config, quoting=csv.QUOTE_ALL)
    wr.writerow(['learning_rate', 'batch_size', 'num_epochs'] + ['training_set_list']*len(training_set_list))
    wr.writerow([learning_rate, batch_size, num_epochs] + list(training_set_dict.keys()))

with open(''.join([model_save_name,'_data','.csv']), w_var) as csv_data:
    wr = csv.writer(csv_data, quoting=csv.QUOTE_ALL)
    wr.writerow(['epoch', 'loss', 'duration [s]'])
    for i in range(0, len(model_loss_list)):
        wr.writerow([model_epoch_list[i], model_loss_list[i], model_time_epoch_list[i]])
