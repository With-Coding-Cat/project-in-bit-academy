import os
import sys
import re
import six
import math
import lmdb
import torch
import cv2

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

class personal_dataset(Dataset):

    def __init__(self, image, opt):
        self.opt = opt
        self.nSamples = 1
        self.image = image

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image

        #img = cv2.resize(img, (self.opt.imgW, self.opt.imgH))

        return (img, 'detected')
