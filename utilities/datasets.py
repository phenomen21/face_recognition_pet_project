import os
import numpy as np
import pandas as pd
import random
import PIL
from PIL import Image
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F
from torchvision import models
from torch.utils.data import Dataset
from collections import deque
from image_processing import crop_img_bbox, align_face
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations
import albumentations.augmentations.transforms as A



class MyRandomHorizontalFlip(torch.nn.Module):
    """
    Adapted from the source of torchvision.transforms.RandomHorizontalFlip
    Returns (image, True) if image was flipped or (image, False) if the image was not flipped
    """

    def __init__(self, p=0.5):
        super().__init__()
        # _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"



class MyCompose:
    """
    Adapted from the source code for torchvision.transforms.Compose for use with MyRandomHorizontalFlip
    Returns tuple (image, True) if image was flipped in HorizontalFlip or (image, False) if the image was not flipped
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
          flipped=False
          if isinstance(t, MyRandomHorizontalFlip):
            img, flipped = t(img)
          else:
            img = t(img)
        return img, flipped

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string



class celebADataset(Dataset):
    '''
    The basic celebA "Faces in the Wild" dataset class used when training the classifier. Takes pandas.DataFrame object with specific structure;
    if bboxes and landmarks are present - crops the image given bbox and processes the face given landmarks.
    Returns a dict structure:
    'images' - batch of cropped and aligned faces from the CelebA dataset of torch.Tensor type, shape given as [batch_size, C, W, H];
    'labels' - array of labels (classes) corresponding with the images;
    'bboxes' - array of bounding boxes, shape as [batch_size, 4], each bounding box;
    'landmarks' - array of landmarks marking 5 points (total 10 numbers) on output images in the following order (as presented in CelebA dataset):
                left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, mouth_left_corner_y, right_mouth_x, mouth_right_corner_y;
    'flipped' - array of boolean numbers indicated if the image was flipped or not;
    'names' - file names of the output images;
    'orig_landmarks' - original landmarks as presented in the CelebA dataset.
    '''
    def __init__(self, data_x, data_y, transform=None, aug=None, align=True, rescale_size=200):
        self.data_x = data_x.copy()
        self.data_y = data_y.copy()
        self.transform = transform
        self.aug = aug
        self.align = align
        self.rescale_size=rescale_size

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bboxes_flag = False
        lndmrks_flag = False
        if len(self.data_x.columns) >= 3:
          bbox = np.array(self.data_x.iloc[idx, 1:5]).astype('float32')
          bboxes_flag = True
        else:
          bbox = np.zeros(4)
        if len(self.data_x.columns) >= 6:
          lndmrk = np.array(self.data_x.iloc[idx,5:]).astype('float32').reshape(-1,2).copy()
          lndmrks_flag = True
        else:
          lndmrk = np.zeros(10).reshape(-1,2)
        orig_lndmrk = lndmrk.copy()
        # try to chop off 10% off the bottom - works well on this specific dataset
        bbox[3] = bbox[3]*0.9
        # crop image given bbox
        img_crp = np.array(crop_img_bbox(self.data_x.iloc[idx, 0],bbox))
        if self.aug:
          img_aug = tt.ToTensor()(self.aug(image=img_crp)['image'])
        else:
          img_aug = tt.ToTensor()(img_crp)
        label = self.data_y.iloc[idx].item()
        imname = self.data_x.iloc[idx, 0]
        # recalculate landmarks given bbox
        if lndmrks_flag:
          lndmrk[:, 0] -= bbox[0]
          lndmrk[:, 1] -= bbox[1]
          lndmrk[:, 0] *= self.rescale_size / (bbox[2] + 1e-10)
          lndmrk[:, 1] *= self.rescale_size / (bbox[3] + 1e-10)
        if self.transform:
          img_crp, flipped = self.transform(img_aug)
        else:
          img_crp = img_aug
          flipped = False
        # flip landmarks
        if lndmrks_flag:
          if flipped:
            lndmrk[:,0] = (self.rescale_size - lndmrk[:,0])
        # align face
        if self.align:
            img_crp, lndmrk = align_face(img_crp,lndmrk)
        # resize
        img_proc = tt.Resize((self.rescale_size, self.rescale_size))(img_proc)
        img_proc = tt.CenterCrop((self.rescale_size, self.rescale_size))(img_proc)
        sample = {'images': img_crp, 'labels':label, 'flipped':flipped,'bboxes':bbox,'landmarks':lndmrk, 'names':imname, 'orig_landmarks':orig_lndmrk}
        return sample



class celebADatasetTriplet(Dataset):
    """
    Dataset for training a classifier using TripletLoss
    Takes a pandas.DataFrame object consisting of three consecutive tables: for anchor image, positive class image and negative class image respectively.
    Returns the same structure as basic CelebADataset class for every type of image (anchor, positive class, negative class)
    """
    def __init__(self, data_x, data_y, transform=None, aug=None, rescale_size=200):
        self.data_x = data_x.iloc[:,:15].copy()
        self.data_y = data_y.copy()
        self.data_pos = data_x.iloc[:,15:31].copy()
        self.data_pos.drop(columns=['label_pos'], inplace=True)
        self.data_neg = data_x.iloc[:,31:].copy()
        self.data_neg.drop(columns=['label_neg'], inplace=True)
        self.transform = transform
        self.aug = aug
        self.rescale_size = rescale_size

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # do the same for the original (anchor) image, positive image and negative image
        postfixes_list = ['','_pos','_neg']
        data_names_list = ['orig', 'pos', 'neg']
        data_list = [self.data_x, self.data_pos, self.data_neg]
        sample = dict()
        label = self.data_y.iloc[idx].item()
        for i,data_src in enumerate(data_list):
            bboxes_flag = False
            lndmrks_flag = False
            if len(data_src.columns) >= 3:
                bbox = np.array(data_src.iloc[idx, 1:5]).astype('float32')
                bboxes_flag = True
            else:
                bbox = np.zeros(4)
            if len(data_src.columns) >= 6:
                lndmrk = np.array(data_src.iloc[idx,5:15]).astype('float32').reshape(-1,2).copy()
                lndmrks_flag = True
            else:
                lndmrk = np.zeros(10).reshape(-1,2)
            orig_lndmrk = lndmrk.copy()
            # try to resize 10% off the bottom - works well on this specific dataset
            bbox[3] = bbox[3]*0.9
            # crop image with bbox
            img_crp = np.array(crop_img_bbox(data_src.iloc[idx, 0],bbox))
            if self.aug:
              img_aug = tt.ToTensor()(self.aug(image=img_crp)['image'])
            else:
              img_aug = tt.ToTensor()(img_crp)

            imname = data_src.iloc[idx, 0]
            # try to recalculate landmarks given bbox
            if lndmrks_flag:
              lndmrk[:, 0] -= bbox[0]
              lndmrk[:, 1] -= bbox[1]
              lndmrk[:, 0] *= self.rescale_size / (bbox[2] + 1e-8)
              lndmrk[:, 1] *= self.rescale_size / (bbox[3] + 1e-8)
            if self.transform:
              img_crp, flipped = self.transform(img_aug)
            else:
              img_crp = img_aug
              flipped = False
            # try to flip landmark
            if lndmrks_flag:
              if flipped:
                lndmrk[:,0] = (self.rescale_size - lndmrk[:,0])
            # align original face
            img_proc, new_lndmrk = align_face(img_crp,lndmrk)
            # resize
            img_proc = tt.Resize((self.rescale_size, self.rescale_size))(img_proc)
            img_proc = tt.CenterCrop((self.rescale_size, self.rescale_size))(img_proc)

            sample['images'+postfixes_list[i]] = img_proc
            sample['labels'+postfixes_list[i]] = label
            sample['bboxes'+postfixes_list[i]] = bbox
            sample['landmarks'+postfixes_list[i]] = new_lndmrk
            sample['names'+postfixes_list[i]] = imname
            sample['orig_landmarks'+postfixes_list[i]] = orig_lndmrk
        return sample



def read_celeba_data(dataset_path='../celeba_dataset'):
  '''
  function to read files coming with CelebA dataset, returns a big pandas dataframe with all the info
  '''

  BB_PATH = os.path.join(dataset_path, 'list_bbox_celeba.txt')
  ID_PATH = os.path.join(dataset_path, 'identity_celeba.txt')
  LM_PATH = os.path.join(dataset_path, 'list_landmarks_celeba.txt')
  bboxes = pd.read_csv(BB_PATH, skiprows=1, delim_whitespace=True)
  labels = pd.read_csv(ID_PATH, sep=' ',header=None,names=['image_id','label'])
  landmarks = pd.read_csv(LM_PATH,skiprows=1,delim_whitespace=True).reset_index().rename(columns={'index':'image_id'})
  data_df = bboxes.merge(labels,on='image_id').merge(landmarks, on='image_id')

  #drop images where nose point is not between the eyes - that means the face is looking far to the right or left
  data_new = data_df[(data_df['lefteye_x'] < data_df['nose_x']) & (data_df['righteye_x'] > data_df['nose_x'])]

  # I will take only those photos whose labels are present LABELS_COUNT_PRESENT times or more in a dataset
  LABELS_COUNT_PRESENT = 20
  label_count = data_df.groupby('label').count()['image_id'].reset_index().rename(columns={'image_id':'count'})

  labels_lim = list(label_count[label_count['count']>=LABELS_COUNT_PRESENT]['label'])

  # I will try to make a small dataset of people that will not be present when training the model
  # will use it for embeddings later
  random.seed(20)
  people_not_present = random.sample(labels_lim, 100) # my 100 people 
  # data_not_present = data_new[data_new['label'].isin(people_not_present)].reset_index()

  labels_lim = list(set(labels_lim) - set(people_not_present))
  data_new2 = data_new[data_new['label'].isin(labels_lim)].reset_index()

  data_new2['label'] = LabelEncoder().fit_transform(data_new2['label'])
  data_new2.drop(columns=[data_new2.columns[0]], inplace=True)
  
  return data_new2



def init_triplet_loaders(dataset_path, stats, batch_size=32, prob=0.5, return_test=False, rescale_size=200):
    '''
    Initializes loaders for TripletLoss, this should be done every epoch so each epoch positive and negative class images would be different
    Takes around 4-5 minutes in free Colab GPU-accelerated platform to initialize (given that it takes 3 hours per epoch to train additional 5 minutes is not a big overhead
    '''
    data = read_celeba_data(dataset_path)

    data1 = data.copy()
    data_pos = data.copy()
    for lab in data1['label'].unique():
        lab_ind = data1[data1['label']==lab].index
        lab_ind2 = deque(lab_ind)
        rand_ind = random.randint(0,10)
        lab_ind2.rotate(rand_ind)
        lab_ind2 = list(lab_ind2)
        data_pos.iloc[lab_ind,:] = data1.iloc[lab_ind2,:]
    data_neg = data.copy()
    for lab in data1['label'].unique():
        lab_ind = data1[data1['label']==lab].index
        negs = data1[data1['label']!=lab].index
        data_neg.iloc[lab_ind,:] = data1.iloc[random.sample(list(negs), len(lab_ind)),:]

    data_pos.columns = ['{}_pos'.format(name) for name in data_pos.columns]
    data_neg.columns = ['{}_neg'.format(name) for name in data_neg.columns]
    data_full = pd.concat([data1, data_pos, data_neg], axis=1)
    X = data_full[data_full.columns[~data_full.columns.isin(['label'])]]
    Y = data_full['label']
    n_classes = Y.nunique()

    x_train, x_testval, y_train, y_testval = train_test_split(X,Y, shuffle=True, stratify=Y,test_size=0.2)
    if return_test:
        x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, shuffle=True, stratify=y_testval, test_size=0.2)

    transform = MyCompose([
          MyRandomHorizontalFlip(0.5),
          tt.Resize((rescale_size, rescale_size)),
          tt.Normalize(*stats)])

    train_transform = albumentations.Compose([
        A.RGBShift(p=prob),
        A.HueSaturationValue(p=prob),
        albumentations.OneOf([
            A.Sharpen(),
            A.AdvancedBlur()], p=prob),
        A.PixelDropout(dropout_prob=0.05, p=np.clip(prob,0,0.3)),
        A.ToSepia(p=np.clip(prob,0,0.2)),
        albumentations.augmentations.dropout.cutout.Cutout(num_holes=1, max_h_size=55, max_w_size=55, p=np.clip(prob,0,0.5)),
        albumentations.augmentations.dropout.cutout.Cutout(num_holes=2, max_h_size=55, max_w_size=55, p=np.clip(prob,0,0.5)),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=3, shadow_dimension=4, p=prob),
        A.MotionBlur(blur_limit=(3,8), p=prob),
        A.CLAHE(clip_limit=10,p=prob),
        A.Posterize(p=prob)])

    train_data = celebADatasetTriplet(x_train,y_train, transform=transform, aug=train_transform)
    if return_test:
        val_data = celebADatasetTriplet(x_val, y_val, transform=transform)
        test_data = celebADatasetTriplet(x_test,y_test, transform=transform)
    else:
        val_data = celebADatasetTriplet(x_testval, y_testval, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    if return_test:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader, n_classes




def init_loaders(dataset_path, stats, batch_size=32, prob=0.5, return_test=True, rescale_size=200, align=True):
  '''
  initialize loaders from dataframe
  returns train val and test loaders and number of classes
  '''
  data1 = read_celeba_data(dataset_path)

  X = data1[data1.columns[~data1.columns.isin(['label','index'])]]
  Y = data1['label']
  x_train, x_testval, y_train, y_testval = train_test_split(X,Y, shuffle=True, stratify=Y,test_size=0.2)
  if return_test:
    x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, shuffle=True, stratify=y_testval, test_size=0.2)
  n_classes = Y.nunique()

  transform = MyCompose([
      MyRandomHorizontalFlip(0.5),
      tt.Resize((rescale_size, rescale_size)),
      tt.Normalize(*stats)])

  train_transform = albumentations.Compose([
    A.RGBShift(p=prob),
    A.HueSaturationValue(p=prob),
    albumentations.OneOf([
        A.Sharpen(),
        A.AdvancedBlur()], p=prob),
    A.PixelDropout(dropout_prob=0.05, p=np.clip(prob,0,0.3)),
    A.ToSepia(p=np.clip(prob,0,0.2)),
    albumentations.augmentations.dropout.cutout.Cutout(num_holes=1, max_h_size=55, max_w_size=55, p=np.clip(prob,0,0.5)),
    albumentations.augmentations.dropout.cutout.Cutout(num_holes=2, max_h_size=55, max_w_size=55, p=np.clip(prob,0,0.5)),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=3, shadow_dimension=4, p=prob),
    A.MotionBlur(blur_limit=(3,8), p=prob),
    A.CLAHE(clip_limit=10,p=prob),
    A.Posterize(p=prob)])

  
  train_data = celebADataset(x_train,y_train, transform, aug=train_transform, align=align, rescale_size=rescale_size)
  if return_test:
    val_data = celebADataset(x_val, y_val, transform, align=align, rescale_size=rescale_size)
    test_data = celebADataset(x_test,y_test, transform, align=align, rescale_size=rescale_size)
  else:
    val_data = celebADataset(x_testval, y_testval, transform, align=align, rescale_size=rescale_size)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)
  if return_test:
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
  else:
    test_loader = None
  return train_loader, val_loader, test_loader, n_classes