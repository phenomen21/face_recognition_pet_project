import os
import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image
import torch.nn as nn
from skimage import io, transform
import matplotlib.patches as patches
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F
from torchvision import models
from torch.utils.data import Dataset
import pickle
import seaborn as sns

import albumentations
import albumentations.augmentations.transforms as A

import timm

stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
RESCALE_SIZE = 200



# suppose that dataset is already downloaded and located in 'celeba_dataset' directory - I intentionally left it empty
# bboxes and landmarks should already be in the dataset folder - files 'list_bbox_celeba.txt', 'list_landmarks_celeba.txt' and '




DATASET_PATH = '../celeba_dataset/'
BB_PATH = os.path.join(DATASET_PATH, 'list_bbox_celeba.txt')
ID_PATH = os.path.join(DATASET_PATH, 'identity_celeba.txt')
LM_PATH = os.path.join(DATASET_PATH, 'list_landmarks_celeba.txt')


