# Face recognition pet-project

This is a pet project in face recognition problem of Computer Vision problem family. The aim of the project was to make a pipeline of face detection, alignment and  recognition without InsightFace-trained neural networks - only neural networks pretrained on ImageNet. The project is developed using PyTorch torchvision framework.

The project started as the final assignment at Deep Learning School (https://en.dlschool.org/) Spring 2022 first semester, the assignment was as follows: 
- train any classifier on "CelebA Aligned and Cropped" dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) using Categorical Cross-Entropy Loss, Triplet Loss and ArcFace Loss (only ImageNet pretrained networks were allowed);
- calculate cosine similarities distributions on pairs of faces that the net never seen before: different pictures of the same person and different people;
- calculate identification rate metric (TPR @ FPR).

The "CelebA Aligned and Cropped" consists of 1k classes (total around 24k pictures). Full CelebA "Faces in the Wild" has over 10k classes and 200k images of people (actual celebrities) with bounding boxes around their faces and landmarks placed on 5 points on each face: right eye, left eye, nose, right mouth corner and left mouth corner.
Example of the pictures with bboxes and landmarks:
![image](https://user-images.githubusercontent.com/89016122/184662355-9353ab9c-81d1-431d-b51a-6a35df813fd9.png)

So the original project was expanded to include big CelebA "Faces in the Wild" dataset with bounding boxes and landmarks, with cropping and aligning big original photos. 
Faces from earler example cropped and aligned using dataset information look like this:
![image](https://user-images.githubusercontent.com/89016122/184663974-e4b337fd-667e-4375-84cc-33f45ebcb99c.png)

Three artificial neural networks were trained separately for this project:
- ImageNet pretrained YOLOv5m (https://github.com/ultralytics/yolov5) was used for face detection. It was transfer trained on CelebA face dataset described above with existing bounding boxes;
- ImageNet pretrained Rexnet_200 network from PyTorch Image Models (timm) (https://github.com/rwightman/pytorch-image-models) was used for landmarks placing, transfer trained on CelebA dataset with existing landmarks;
- Another ImageNet pretrained Rexnet_200 neural network from timm was used for classification and embedding calculation, also was transfer trained on the CelebA dataset using CCE and ArcFace losses.

All the training was performed on free Google Colab platform with GPU acceleration. The Albumentation library (https://albumentations.ai/) was used to augment pictures on-the-fly. The code used for training neural networks, augmenting photos, aligning and cropping faces is presented in this repository. The dataset CelebA is available on its official website https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and is not included in this repository. The training process is described in details in /training/README.md

A simple web-application is built using Flask-RESTful that demonstrates the face recognition pipeline on an uploaded custom photo of a person:
![image](https://user-images.githubusercontent.com/89016122/185673445-a052346b-ea51-4402-a3c5-12375cb12ef5.png)

The app also searches the celebrity whose face is most similar (by means of cosine distance between corresponding embeddings) to the person on the photo provided.
