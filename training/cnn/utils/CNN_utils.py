from __future__ import print_function, division
import os
import random
import torch
import math
import numbers
import pandas as pd
from skimage import io, transform
from PIL import Image
from PIL import ImageEnhance
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# TODO: Make it possible to work with batches
class LanePoseDataset(Dataset):
    def __init__(self, csvPath, csvFilename, imgPath, transform=None):
        """
        Args:
            csvPath (string): Path to the csv file.
            csvFilename (string): Filename of the csv file with extension.
            imgPath (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        self.data = pd.read_csv("".join([csvPath, csvFilename]), header=0)
        self.imgPath = imgPath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data.iloc[idx, 0]
        if '.jpg' not in image_name:
            image_name = image_name + '.jpg'
        img_name = os.path.join(self.imgPath, image_name)



        # image = io.imread(img_name,as_gray=True)
        image = io.imread(img_name)

        pose = self.data.iloc[idx, 2:4]
        pose = np.array(pose)
        pose = pose.astype('float')

        if self.transform is not None:
            # torchvision transforms
            image = Image.fromarray(image)
            image = self.transform(image)

            # custom transforms

        return image, pose

class TransCropHorizon(object):
    """Crop the Horizon.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, crop_value, set_black=False):
        assert isinstance(set_black, (bool))
        self.set_black = set_black

        if crop_value >= 0 and crop_value < 1:
            self.crop_value = crop_value
        else:
            print('One or more Arg is out of range!')

    def __call__(self, image):
        crop_value = self.crop_value
        set_black = self.set_black
        image_heiht = image.size[1]
        crop_pixels_from_top = int(round(image_heiht*crop_value,0))

        # convert from PIL to np
        image = np.array(image)

        if set_black==True:
            image[:][0:crop_pixels_from_top-1][:] = np.zeros_like(image[:][0:crop_pixels_from_top-1][:])
        else:
            image = image[:][crop_pixels_from_top:-1][:]

        # plt.figure()
        # plt.imshow(image)
        # plt.show()  # display it

        # convert again to PIL
        image = Image.fromarray(image)

        return image



class TransConvCoord(object):
    """Apply the ConvCoord sys to the image.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, image):
        image = np.array(image) # input image is Image(mode:L)
        # print(image.shape)
        if len(image.shape) == 2: # if Grayscale
            image_h, image_w = image.shape[0:2]
            image_ch = 1
        else:  # if RGB
            image_h, image_w, image_ch = image.shape[0:3]


        convCh_tb = np.zeros([image_h, image_w])
        convCh_lr = np.zeros([image_h, image_w])

        for i_tb in range(0,image_h):
            for j_tb in range(0,image_w):
                convCh_tb[i_tb, j_tb] = (i_tb)/(image_h-1)

        for i_lr in range(0,image_h):
            for j_lr in range(0,image_w):
                convCh_lr[i_lr, j_lr] = (j_lr)/(image_w-1)

        image_new = np.zeros([image_h, image_w, image_ch+2])

        if image_ch == 1:
            image_new[:,:,0] = image
            image_new[:,:,1] = convCh_tb
            image_new[:,:,2] = convCh_lr
        if image_ch == 3:
            image_new[:,:,0:3] = image
            image_new[:,:,3] = convCh_tb
            image_new[:,:,4] = convCh_lr

        return image_new


class RandImageAugment(object):
    """Crop the Horizon.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, augment_white_balance = True, white_balance_augment_max_deviation = 0.6, white_balance_augment_sigma = 0.2, augment_brightness = True, brightness_augment_max_deviation = 0.6, brightness_augment_sigma = 0.2, augment_contrast = True, contrast_augment_max_deviation = 0.6, contrast_augment_sigma = 0.2):
        assert isinstance(augment_white_balance, (bool))
        assert isinstance(augment_brightness, (bool))
        assert isinstance(augment_contrast, (bool))

        self.augment_white_balance = augment_white_balance
        self.white_balance_augment_max_deviation = white_balance_augment_max_deviation
        self.white_balance_augment_sigma = white_balance_augment_sigma

        self.augment_brightness = augment_brightness
        self.brightness_augment_max_deviation = brightness_augment_max_deviation
        self.brightness_augment_sigma = brightness_augment_sigma

        self.augment_contrast = augment_contrast
        self.contrast_augment_max_deviation = contrast_augment_max_deviation
        self.contrast_augment_sigma = contrast_augment_sigma

    def __call__(self, image):
        augment_white_balance = self.augment_white_balance
        white_balance_augment_max_deviation = self.white_balance_augment_max_deviation
        white_balance_augment_sigma = self.white_balance_augment_sigma

        augment_brightness = self.augment_brightness
        brightness_augment_max_deviation = self.brightness_augment_max_deviation
        brightness_augment_sigma = self.brightness_augment_sigma

        augment_contrast = self.augment_contrast
        contrast_augment_max_deviation = self.contrast_augment_max_deviation
        contrast_augment_sigma = self.contrast_augment_sigma

        if augment_white_balance:
            image = np.array(image)

            if image.shape[2] == 3:
                # Produce Gaussian distribution for R,G,B
                gauss_R = abs(random.gauss(0, white_balance_augment_sigma))
                gauss_G = abs(random.gauss(0, white_balance_augment_sigma))
                gauss_B = abs(random.gauss(0, white_balance_augment_sigma))

                # Check Gauss R
                if gauss_R < white_balance_augment_max_deviation:
                    R_w = 255*(1+gauss_R)
                else:
                    R_w = 255

                # Check Gauss G
                if gauss_G < white_balance_augment_max_deviation:
                    G_w = 255*(1+gauss_G)
                else:
                    G_w = 255

                # Check Gauss B
                if gauss_B < white_balance_augment_max_deviation:
                    B_w = 255*(1+gauss_B)
                else:
                    B_w = 255

                RGB_w = np.transpose([255/R_w, 255/G_w, 255/B_w])
                I_RGB_w = np.identity(3)*RGB_w

                for h in range(0,image.shape[0]-1):
                    for w in range(0,image.shape[1]-1):
                        image[h,w] = np.matmul(I_RGB_w, image[h,w])

            image = Image.fromarray(image)


        if augment_brightness:
            augmenter_brightness = ImageEnhance.Brightness(image)

            factor = random.gauss(1, brightness_augment_sigma)
            if factor > 1-brightness_augment_max_deviation and factor < 1+brightness_augment_max_deviation:
                image = augmenter_brightness.enhance(factor)
                # if out of range do nothing!

        if augment_contrast:
            augmenter_contrast = ImageEnhance.Contrast(image)

            factor = random.gauss(1, contrast_augment_sigma)
            if factor > 1-contrast_augment_max_deviation and factor < 1+contrast_augment_max_deviation:
                image = augmenter_contrast.enhance(factor)
                # if out of range do nothing!

        # image.show()
        return image

class ToCustomTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, use_convcoord):
        self.use_convcoord = use_convcoord


    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        # handle numpy arrays
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            if self.use_convcoord:
                pic[:,:,0] = pic[:,:,0]/255
            else:
                pic = pic/255

            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

class TransRandomErasing(transforms.RandomErasing):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super(TransRandomErasing, self).__init__(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = random.randint(0, 255)/255
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

    def __call__(self, img):
        super(TransRandomErasing, self).__call__(img)



# Helper function to show a batch
def showBatch(images_batch, poses_batch, as_grayscale, batch_size, outputs=10):
    fig=plt.figure(figsize=(12, 8))
    columns = 4
    rows = max([round(batch_size/columns, 0),1])
    for i, image in enumerate(images_batch):
        image = np.array(image)
        fig.add_subplot(rows, columns, i+1)
        if as_grayscale:
            plt.imshow(image[0], cmap='gray', vmin=0, vmax=1)
        else:
            image = image.transpose((1, 2, 0))
            plt.imshow(image, vmin=0, vmax=1)
        plt.text(0, 0, ''.join(['d: ', str(round(poses_batch[i][0].item(),4)), ' theta: ', str(round(poses_batch[i][1].item(),2))]))
        if outputs != 10:
            plt.text(0, -10, ''.join(['d: ', str(round(outputs[i][0].item(),4)), ' theta: ', str(round(outputs[i][1].item(),2))]))
    plt.show()
