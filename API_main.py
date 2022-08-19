import io
import os
from flask import Flask, make_response, request, render_template_string, render_template
from flask_restful import Resource, Api, reqparse
import werkzeug
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F
import cv2
import numpy as np
from yolov5 import *
# import yolov5.models as models
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from yolov5.utils.general import non_max_suppression
from utilities.image_processing import crop_img_bbox, align_face, save_img
import zipfile


RESCALE_SIZE = 200
MODELS_DIR='./models/'
PROC_DIR = './processing/'
PHOTOS_DIR = './photos/'

# load models
model_yolov5 = DetectMultiBackend(os.path.join(MODELS_DIR, 'yolov5m_detect.pt'), device='cpu', dnn=False, data=os.path.join(MODELS_DIR, 'celeba.yaml'), fp16=False)
model_landm = torch.load(os.path.join(MODELS_DIR, 'rexnet_landmarks.pt'), map_location='cpu')
model_emb = torch.load(os.path.join(MODELS_DIR, 'rexnet_200_arc.pt'), map_location='cpu')

# load embeddings and photos
full_embeddings = np.load(os.path.join(MODELS_DIR, 'embeddings.npy'))
full_labels = np.load(os.path.join(MODELS_DIR, 'labels.npy'),allow_pickle=True)
if not os.path.exists(PHOTOS_DIR):
    with zipfile.ZipFile('photos.zip', 'r') as zip_ref:
        zip_ref.extractall('')



def detect_face(image_path, model_detect):
    img0 = cv2.imread(image_path)  # BGR
  #  assert img0 is not None, f'Image Not Found {path}'
  # Padded resize
    img = letterbox(img0, 640, stride=model_detect.stride, auto=True)[0]
  # Convert
    img = np.moveaxis(img,2,0)[...,::-1]  # HWC to CHW, BGR to RGB
  # img = np.moveaxis(img,2,0)  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    output = model_detect(img)
    pred = non_max_suppression(output, 0.25, 0.45, None, False, max_det=1000)
    rat_w = img0.shape[0] / img.shape[2]
    rat_h = img0.shape[1] / img.shape[3]
    pred = pred[0][0]
    pred[2] = (pred[2]-pred[0])
    pred[3] = pred[3]-pred[1]
    pred_sc = pred.clone()
    pred_sc[0] = img0.shape[1] - (pred[0] + pred[2]) * rat_h
    pred_sc[1] = pred[1] * rat_w
    pred_sc[2] = pred[2]* rat_h * 0.9
    pred_sc[3] = pred[3] * rat_w
    print(image_path)
    save_img(image_path,bbox=pred_sc,img_path='',file_name='face_detected')
    img_cropped = crop_img_bbox(image_path, pred_sc)
    return img_cropped

def place_landmarks(img_cropped, model_landmarks):
    img_cropped_proc = tt.ToTensor()(img_cropped)
    img_cropped_proc = tt.Resize((RESCALE_SIZE, RESCALE_SIZE))(img_cropped_proc)[None]
    landmarks = model_landmarks(img_cropped_proc).detach()
    # show_img(img_cropped_proc, landmarks=landmarks,ax=axes[2],name='Landmarks created')
    # align face
    save_img(img_cropped_proc,landmarks=landmarks,file_name='landmarks')
    return img_cropped_proc

def align(img_cropped_proc, landmarks):
    img_aligned, new_landmarks = align_face(img_cropped_proc, landmarks[0].reshape(5,2))
    img_al_proc = tt.Resize((RESCALE_SIZE, RESCALE_SIZE))(img_aligned)
    img_al_proc = tt.CenterCrop((RESCALE_SIZE, RESCALE_SIZE))(img_al_proc)
    # show_img(img_al_proc, ax=axes[3], name='Face aligned')
    save_img(img_al_proc,file_name='face_aligned')
    return img_al_proc

def find_celeb(image_file):

    return

class ProcessImage(Resource):
    
    def post(self):
        for file in os.listdir(PROC_DIR):
            print(file)
            os.remove(os.path.join(PROC_DIR, file))
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', required=True)  # add args

        args = parser.parse_args()
        image_file = args['file']

        orig_image=os.path.join(PROC_DIR,  "orig_image.jpg")
        image_file.save(orig_image)

        # stage 1 - detect a face
        img_cropped = detect_face(orig_image, model_yolov5)

        # stage 2 - place landmarks 
        img_land, landmarks = place_landmarks(img_cropped, model_landm)

        # stage 3 - align (no neural network here it just looks interesting)
        aligned = align(img_land, landmarks)

        # stage 4 - show celebs
        celeb = find_celeb(aligned)


        with io.open('results.html', 'r', encoding="utf-8") as index:
            page = index.read()
        return make_response(render_template_string(page))
        return   

    def get(self):
        with io.open('index.html', 'r', encoding="utf-8") as index:
            page = index.read()
        return make_response(render_template_string(page))


app = Flask(__name__)
api = Api(app)
api.add_resource(ProcessImage, '/')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6000)), debug=True)