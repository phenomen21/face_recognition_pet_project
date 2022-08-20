import io
import os
import shutil
from flask import Flask, make_response, redirect, render_template_string, render_template, url_for
from flask_restful import Resource, Api, reqparse
import werkzeug
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F
import cv2
import numpy as np
import pandas as pd
from yolov5 import *
# import yolov5.models as models
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from yolov5.utils.general import non_max_suppression
from utilities.image_processing import crop_img_bbox, align_face, save_img
import zipfile


RESCALE_SIZE = 200
MODELS_DIR='models'
PROC_DIR = 'static'
PHOTOS_DIR = 'photos'
DEVICE = torch.device('cpu')

# load models
model_yolov5 = DetectMultiBackend(os.path.join(MODELS_DIR, 'yolov5m_detect.pt'), device=DEVICE, dnn=False, data=os.path.join(MODELS_DIR, 'celeba.yaml'), fp16=False)
model_landm = torch.load(os.path.join(MODELS_DIR, 'rexnet_landmarks.pt'), map_location=DEVICE)
model_emb = torch.load(os.path.join(MODELS_DIR, 'rexnet_200_arc.pt'), map_location=DEVICE)

# load embeddings and photos
photo_list = pd.read_csv(os.path.join(MODELS_DIR, 'photo_list.csv'),index_col=0)
full_embeddings = np.load(os.path.join(MODELS_DIR, 'embeddings.npy'))
full_labels = np.load(os.path.join(MODELS_DIR, 'labels.npy'),allow_pickle=True)
if not os.path.exists(PHOTOS_DIR):
    with zipfile.ZipFile('photos.zip', 'r') as zip_ref:
        zip_ref.extractall('')



def detect_face(image_path, model_detect):
    img0 = cv2.imread(image_path)  # BGR
  #  assert img0 is not None, f'Image Not Found {path}'
  # Padded resize
    img = letterbox(img0, 1024, stride=model_detect.stride, auto=True)[0]
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
    pred_sc[2] = pred[2]* rat_h 
    pred_sc[3] = pred[3] * rat_w * 0.9
    print(image_path)
    save_img(image_path, bbox=pred_sc, name='detected.jpg', save_path=PROC_DIR)
    img_cropped = crop_img_bbox(image_path, pred_sc, IMG_PATH=PROC_DIR)
    return img_cropped

def place_landmarks(img_cropped, model_landmarks):
    img_cropped_proc = tt.ToTensor()(img_cropped)
    img_cropped_proc = tt.Resize((RESCALE_SIZE, RESCALE_SIZE))(img_cropped_proc)[None]
    landmarks = model_landmarks(img_cropped_proc).detach()
    save_img(img_cropped_proc, landmarks=landmarks, name='landmarks.jpg', save_path=PROC_DIR)
    return img_cropped_proc, landmarks

def align(img_cropped_proc, landmarks):
    img_aligned, new_landmarks = align_face(img_cropped_proc, landmarks[0].reshape(5,2))
    img_al_proc = tt.Resize((RESCALE_SIZE, RESCALE_SIZE))(img_aligned)
    img_al_proc = tt.CenterCrop((RESCALE_SIZE, RESCALE_SIZE))(img_al_proc)
    save_img(img_al_proc, name='face_aligned.jpg', save_path=PROC_DIR)
    return img_al_proc

def find_celeb(image, model_emb):
    # make embeddings
    embedding = model_emb(image).detach()[0]

    # calculate cosine similarities
    tensor_embeddings = torch.Tensor(full_embeddings)
    cosines = torch.nn.functional.cosine_similarity(embedding,tensor_embeddings)
    # choose the label which corresponds to the highest similarity
    best_label = full_labels[cosines.argmax().numpy()]
    best_image = photo_list[photo_list['label']==best_label]['image_id'].item() # filename
    src_path = os.path.join(PHOTOS_DIR, best_image)
    dst_path = os.path.join(PROC_DIR,'celeb.jpg')
    shutil.copy(src_path,dst_path)
    return dst_path

class ProcessImage(Resource):
    
    def post(self):
        if not os.path.exists(PROC_DIR):
            os.mkdir(PROC_DIR)
        for file in os.listdir(PROC_DIR):
            if file.endswith('jpg'):
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
        celeb = find_celeb(aligned, model_emb)

        # with io.open('results.html', 'r', encoding="utf-8") as index:
        #     page = index.read()
        # return make_response(render_template_string(page))
        return redirect(url_for('showresults'))

    def get(self):
        with io.open('index.html', 'r', encoding="utf-8") as index:
            page = index.read()
        # page = page.format(orig_image=os.path.abspath('processing/orig_image.jpg'))
        return make_response(render_template_string(page))

class ShowResults(Resource):
    def get(self):
        with io.open('results.html', 'r', encoding="utf-8") as index:
            page = index.read()

        import pathlib
        # print('current dir:', pathlib.Path().resolve())
        # orig = os.path.join(PROC_DIR, 'orig_image.jpg')
        return make_response(render_template_string(page))
    
    def post(self):
        print('redirect')
        ProcessImage.post(self)


app = Flask(__name__)
api = Api(app)
api.add_resource(ProcessImage, '/')
api.add_resource(ShowResults, '/results')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6000)), debug=True)