import io
import os
from flask import Flask, make_response, request, render_template_string, render_template
from flask_restful import Resource, Api, reqparse
import werkzeug
import torch
import yolov5
import yolov5.models as models
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from yolov5.utils.general import non_max_suppression
from .utils.image_processing import crop_img_bbox, align_face




WORK_DIR = './processing/'


def save_detection(image_file):

    return

def save_landmarked(image_file):

    return

def save_aligned(image_file):

    return

def find_celeb(image_file):

    return

class ProcessImage(Resource):
    
    def post(self):
        for file in os.listdir(WORK_DIR):
            print(file)
            os.remove(os.path.join(WORK_DIR, file))
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', required=True)  # add args

        args = parser.parse_args()
        image_file = args['file']

        orig_image=os.path.join(WORK_DIR,  "orig_image.jpg")
        image_file.save(orig_image)


        # stage1 - detect
        detected = save_detection(orig_image)

        # stage2 - landmarks
        landmarked = save_landmarked(detected)

        # stage3 - align
        aligned = save_aligned(landmarked)

        # stage4 - show celebs
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