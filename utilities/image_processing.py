import os
import cv2
import numpy as np
import math
import PIL
from PIL import Image
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def show_img(image_name, bbox=None, landmarks=None, ax=None, name='', img_path='celeba_dataset'):
    '''
    for one image
    works both for file names, paths and for images in dataset,
    also shows bboxes if present, also shows landmarks if present
    '''
    if ax is None:
      f, ax= plt.subplots(1,1,figsize=(5,5))
      # print(type(img))
    if isinstance(image_name,str):
      if os.path.isabs(image_name):
        img = Image.open(image_name)
      else:
        img = Image.open(os.path.join(img_path, image_name))
      ax.imshow(img, cmap='gray')
    elif isinstance(image_name, torch.Tensor):
      img = torch.permute(torch.squeeze(image_name), (1,2,0))
      # img = denorm(img)
      # print(img.shape)
      ax.imshow(img)
    elif isinstance(image_name,PIL.Image.Image):
      ax.imshow(image_name)

    if bbox is not None:
      rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=3, edgecolor='black', facecolor='none')
      ax.add_patch(rect)
    if landmarks is not None:
      landmrks = np.array(landmarks).reshape(-1, 2)
      ax.scatter(landmrks[:, 0], landmrks[:, 1], s=30, marker='*', c='r')
    ax.set_title(name)
    # plt.show()



def cv_draw_star(image, x,y,size):
  phi = 4 * np.pi / 5
  rotations = [[[np.cos(i * phi), -np.sin(i * phi)], [i * np.sin(phi), np.cos(i * phi)]] for i in range(1, 5)]
  star = np.array([[[[0, -1]] + [np.dot(m, (0, -1)) for m in rotations]]], dtype=float)
  shift_star = np.round(star * size + np.array([x, y])).astype(int)
  return cv2.polylines(image, shift_star, True, (255, 0, 0), 2)



def save_img(image, bbox=None, landmarks=None, name='/content/image.jpg'):
    '''
    for one image
    works both for images names and for images in dataset, also saves bboxes if present
    also shows landmarks if present
    '''
    if isinstance(image, str):
      img = cv2.imread(image)
    elif isinstance(image, torch.Tensor):
      img = cv2.cvtColor(np.ascontiguousarray(image[0].numpy().transpose(1,2,0))*255, cv2.COLOR_RGB2BGR)
    elif isinstance(image, PIL.Image.Image):
      img = np.ascontiguousarray(np.array(image).transpose(1,2,0))

    if bbox is not None:
      bbox=bbox.int().numpy()
      # print(bbox)
      start = (bbox[0], bbox[1])
      end = (bbox[0]+bbox[2], bbox[1]+bbox[3])
      color = (0,0,255)
      img = cv2.rectangle(img, start, end, color, 3)
    if landmarks is not None:
      landmrks = np.array(landmarks).reshape(-1, 2).astype(int)
      for lnd in landmrks:
        img = cv_draw_star(img,lnd[0],lnd[1],3)
    cv2.imwrite(name,img)  



def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]



def show_imgs(images_names, bboxes=None, landmarks=None, num=5, img_path='celeba_dataset'):
    '''
    works both for file names, paths and for images in dataset, also shows bboxes if present
    also shows landmarks if present
    '''
    i=0
    f, axes= plt.subplots(1, min(num,len(images_names)), figsize=(3*min(num,len(images_names)),num))
    for i, axis in enumerate(axes):

      img = images_names[i]
      # print(type(img))
      if isinstance(img,str):
        if os.path.isabs(img):
          img = Image.open(img)
        else:
          img = Image.open(os.path.join(img_path, img))
      elif isinstance(img, torch.Tensor):
        img = torch.permute(img, (1,2,0))
        img = denorm(img)
      axes[i].imshow(img, cmap='gray')

      if bboxes is not None:
        rect = patches.Rectangle((bboxes[i][0], bboxes[i][1]), bboxes[i][2], bboxes[i][3], linewidth=2, edgecolor='black', facecolor='none')
        axes[i].add_patch(rect)
      if landmarks is not None:
        landmrks = np.array(landmarks)[i].reshape(-1, 2)
        axes[i].scatter(landmrks[:, 0], landmrks[:, 1], s=20, marker='.', c='r')
      if i == num:
        break
    plt.show()


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach().cpu()[:nmax]), nrows=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for batch in dl:
      show_images(batch['images'], nmax)
      break


def crop_img_bbox(image, bbox, IMG_PATH='celeba_dataset'):
  '''
  crop image given bbox, takes PIL image or torch.Tensor
  returns cropped image of the same type
  '''
  if isinstance(bbox, torch.Tensor):
    bbox = bbox.numpy()
  if isinstance(image, str): #image = filename
    if os.path.isabs(image):
      img = Image.open(image)
    else:
      img = Image.open(os.path.join(IMG_PATH, image))
    if not np.allclose(bbox,np.zeros_like(bbox)):
      img_cropped = img.crop((bbox[0], bbox[1],bbox[0]+bbox[2], bbox[1]+bbox[3]))
    else:
      img_cropped = img
  else:
    if not np.allclose(bbox,np.zeros_like(bbox)):
      img_cropped = tt.functional.crop(image, bbox[1],bbox[0], bbox[3], bbox[2])
    else:
      img_cropped = image
  return img_cropped


def align_face(image, landmarks):
  '''
  put eyes on the same line and nose on the same line somewhere below the eye-line
  overall three points of interests:
  eyeline will be 45% off the upper border and centered
  nose will be at the center and 60% below the upper border
  function takes torch.Tensor
  '''
  # find the rotation center and rotate - rotate around nose
  if not np.allclose(landmarks, np.zeros_like(landmarks)):
    right = landmarks[0]
    left = landmarks[1]
    tg_angle = (left[1] - right[1]) / (right[0] - left[0])
    angle = math.degrees(np.arctan(tg_angle))
    nose_point = list(landmarks[2])
    new_image = F.rotate(image, -angle, expand=False,center = nose_point)
    # rotate landmarks themselves
    new_landmarks = np.zeros_like(landmarks)
    r_angle = math.radians(angle)
    new_landmarks[:, 0] = (landmarks[:, 0] - nose_point[0])*np.cos(r_angle) - (landmarks[:, 1] - nose_point[1])*np.sin(r_angle) + nose_point[0]
    new_landmarks[:, 1] = (landmarks[:, 0] - nose_point[0])*np.sin(r_angle) + (landmarks[:, 1] - nose_point[1])*np.cos(r_angle) + nose_point[1]

    # move the face so eyes will be on 4% mark
    EYE_LINE = 0.4
    MOUTH_LINE = 0.75
    width = round(abs(new_landmarks[0,0]-new_landmarks[1,0])*2)
    top_left_x = round(min(new_landmarks[0,0],new_landmarks[1,0]) - width//4)
    height = round((new_landmarks[3,1]-new_landmarks[0,1]) / (MOUTH_LINE - EYE_LINE))
    top_left_y = round(new_landmarks[0,1] - height*EYE_LINE)

    # we have now a bbox, when rescaling it we will have desired features on desired lines (probably)
    # now we need to crop
    new_image = F.crop(new_image, top_left_y, top_left_x,  height, width)
  else:
    new_image = image
    new_landmarks = landmarks
  return new_image, new_landmarks
