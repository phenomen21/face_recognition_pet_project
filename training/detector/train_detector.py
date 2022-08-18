import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# clone YOLOv5 repo
os.system('git clone https://github.com/ultralytics/yolov5')

# install requirements
os.system('pip install -r yolov5/requirements.txt')




# suppose that dataset is already downloaded and located in 'celeba_dataset' directory - I intentionally left it empty
# bboxes and landmarks should already be in the dataset folder - files 'list_bbox_celeba.txt' and 'list_landmarks_celeba.txt'

DATASET_PATH = '../celeba_dataset/'
BB_PATH = os.path.join(DATASET_PATH, 'list_bbox_celeba.txt')
ID_PATH = os.path.join(DATASET_PATH, 'identity_celeba.txt')
LM_PATH = os.path.join(DATASET_PATH, 'list_landmarks_celeba.txt')

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

data_new2 = data_new[data_new['label'].isin(labels_lim)].reset_index()

data_new2['label'] = LabelEncoder().fit_transform(data_new2['label'])
X = data_new2[data_new2.columns[~data_new2.columns.isin(['label','index'])]]
Y = data_new2['label']
x_train, x_testval, y_train, y_testval = train_test_split(X,Y, shuffle=True, stratify=Y,test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, shuffle=True, stratify=y_testval, test_size=0.5)
n_classes = data_new2['label'].nunique()

# we need to recalculate bboxes from CelebA dataset to COCO format - each image should have a specific txt file with coordinates of bounding boxes
# required format is "center_x, center_y, width, height" in the percentage of image dimensions
# so we create the 'labels' folder where these text files will be located
LAB_PATH = 'labels'
if not os.path.exists(LAB_PATH):
  os.system('mkdir labels')
  for i, image_name in enumerate(data_new2['image_id']):
    img = Image.open(os.path.join(DATASET_PATH, image_name))
    im_width = img.size[0]
    im_height = img.size[1]
    x_center = (data_new2.iloc[i,2] + data_new2.iloc[i, 4] / 2) / im_width
    y_center = (data_new2.iloc[i,3] + data_new2.iloc[i, 5] / 2) / im_height
    bb_w = data_new2.iloc[i, 4] / im_width
    bb_h = data_new2.iloc[i, 5] / im_height
    f_name = os.path.join(LAB_PATH, image_name.replace('jpg','txt'))
    with open(f_name, 'w') as f:
      f.write('0 {c_x:.6f} {c_y:.6f} {w:.6f} {h:.6f}'.format(c_x=x_center, c_y=y_center, w=bb_w, h=bb_h))


# now we need to move all the images to their corresponding folders - "train" "val" and "test"
os.system('mkdir images/train images/val images/test labels/train labels/val labels/test')
for train_image in list(x_train['image_id']):
  os.rename(os.path.join(DATASET_PATH, train_image), os.path.join(DATASET_PATH,'train',train_image))
  os.rename(os.path.join(LAB_PATH, train_image.replace('jpg', 'txt')), os.path.join(LAB_PATH,'train',train_image.replace('jpg', 'txt')))
for val_image in list(x_val['image_id']):
  os.rename(os.path.join(DATASET_PATH, val_image), os.path.join(DATASET_PATH,'val',val_image))
  os.rename(os.path.join(LAB_PATH, val_image.replace('jpg', 'txt')), os.path.join(LAB_PATH,'val',val_image.replace('jpg', 'txt')))
for test_image in list(x_test['image_id']):
  os.rename(os.path.join(DATASET_PATH, test_image), os.path.join(DATASET_PATH,'test',test_image))
  os.rename(os.path.join(LAB_PATH, test_image.replace('jpg', 'txt')), os.path.join(LAB_PATH,'test',test_image.replace('jpg', 'txt')))

# now run training script
os.system('yolov5/train.py --img 1024 --cfg yolov5m.yaml --hyp hyp.celeba.yaml --batch 15 --epochs 3 --data celeba.yaml --weights yolov5m.pt --workers 2 --name yolo_celeba')
