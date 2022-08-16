# Face recognition pet-project

This is a pet project in face recognition problem of Computer Vision problem family. The aim of the project was to make a pipeline of face detection, alignment and  recognition without InsightFace-trained neural networks - only neural networks pretrained on ImageNet. The project is developed using PyTorch torchvision framework.

The project arose from the final assignment at Deep Learning School (https://en.dlschool.org/) Spring 2022 first semester, the assignment was as follows: 
- train any classifier on "CelebA Aligned and Cropped" dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) using Categorical Cross-Entropy Loss, Triplet Loss and ArcFace Loss (only ImageNet pretrained networks were allowed);
- calculate cosine similarities distributions on pairs of faces that the net never seen before: different pictures of the same person and different people;
- calculate identification rate metric (TPR @ FPR).
The "CelebA Aligned and Cropped" consists of 1k classes (total around 24k pictures). Full CelebA "Faces in the Wild" has over 10k classes and 200k images of people (actual celebrities) with bounding boxes around their faces and landmarks placed on 5 points on each face: right eye, left eye, nose, right mouth corner and left mouth corner.
Example of the pictures with bboxes and landmarks:
![image](https://user-images.githubusercontent.com/89016122/184662355-9353ab9c-81d1-431d-b51a-6a35df813fd9.png)

So the original project was expanded to include big CelebA "Faces in the Wild" dataset with bounding boxes and landmarks, with cropping and aligning. 
Cropped and aligned faces look like this:
![image](https://user-images.githubusercontent.com/89016122/184663974-e4b337fd-667e-4375-84cc-33f45ebcb99c.png)

Three artificial neural networks were trained separately for this project:
- ImageNet pretrained Rexnet_200 neural network from PyTorch Image Models (timm) (https://github.com/rwightman/pytorch-image-models) was used for classification and embedding calculation. The network was transfer trained on the CelebA faces dataset described above using ArcFace Loss.
- ImageNet pretrained YOLOv5m (https://github.com/ultralytics/yolov5) was used for face detection, also transfer trained on CelebA dataset with bounding boxes.
- Another ImageNet pretrained Rexnet_200 network from timm was used for landmarks placing, it was transfer trained on CelebA dataset with landmarks.

All the training was performed on free Google Colab platform with GPU acceleration. The Albumentation library (https://albumentations.ai/) was used to augment pictures on-the-fly. The code used for training neural networks, augmenting photos, aligning and cropping faces is presented in this repository. The dataset CelebA is available on their official website https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and is not included in this repository.

The next step is to build a web-application that would demonstrate the pipeline as follows:
![image](https://user-images.githubusercontent.com/89016122/184847603-f1e2c0cb-f699-411a-b1cf-6da11eabad7a.png)
An application will be built using Flask-RESTful
