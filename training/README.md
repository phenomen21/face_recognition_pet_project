The **face recognition task** consists of several sub-procedures:
- face detection
- face alignment
- face classification (recognition)

For each of these sub-tasks a separate artificial neural network was trained using functions presented in training.py. Training was performed on Google Colab free platform with GPU acceleration.

For every network the CelebA "Faces in the Wild" dataset was cleaned in the way that only pictures with faces whose nose was located somewhere between the eyes were used. Also only classes that have 20 or more pictures in them were selected. Additionally 100 classes were removed from the dataset for test purposes. The resulting dataset consists of 6248 classes and has 109433 different pictures.


1. **Detection**

ImageNet pretrained YOLOv5m from https://github.com/ultralytics/yolov5 was transfer trained on cleaned CelebA dataset with bounding boxes to detect a face. The original training script from the creators of YOLOv5 requires one parameter file specifying detectable classes and one hyperparameter file specifying hyperparameters used in training. The parameter file celeba.yaml specifying exactly one detectable class and hyperparameter file hyp.celeba.yaml were used in the training script included in the original repository. Pictures from the training dataset looked like this:

![image](https://user-images.githubusercontent.com/89016122/185159905-04ae0edd-0491-4940-a274-bbd38be763da.png)

- learning rate was set to 1e-3;
- 2 epochs were completed;
- each training epoch took approximately 2h 40min to complete;
- validation step after each epoch of training lasted 10min;
- after every epoch the Google Colab session had to be terminated and moved to another free account;

|epoch | precision| recall |  mAP@.5|
|-----|---------|--------|---------|
|  1   |    0.981 | 0.989  |  0.993 |
|  2   |    0.986 | 0.994  |  0.994 |
-------------------------------------------

2. **Landmarks**

ImageNet pretrained Rexnet_200 from PyTorch Image Models repository (timm - https://github.com/rwightman/pytorch-image-models) was transfer trained on cleaned CelebA dataset with landmarks. Albumentation libaray (https://albumentations.ai/) was used to regularize training process by augmenting images. Pictures from the training dataset looked like this:
![image](https://user-images.githubusercontent.com/89016122/185155253-2c154e42-9a94-462d-a2b1-af9440396d80.png)

- learning rates were set to 1e-4 for the feature extractor and 1e-3 for the head of the model;
- 3 epochs were completed;
- each training epoch took approximately 45min to complete;
- validation step took 5min;

|epoch |  train_loss  |  val_loss|
|---|---|---|
|  1   |    235.64  |       46.72 |
|  2   |     63.94  |       42.19 |
|  3   |     52.59  |       43.49 |

_#TODO: increase the quality of landmarks placing; try different neural networks, use better augmentation_

---------------------------------------

3. **Classifier**

ImageNet pretrained Rexnet_200 from PyTorch Image Models repository (timm - https://github.com/rwightman/pytorch-image-models) was transfer trained on cleaned CelebA dataset where images were cropped and aligned using bboxes and landmarks already existed in the dataset. Albumentation libaray (https://albumentations.ai/) was used to regularize training process by augmenting images. Pictures from the training dataset looked like this:
![image](https://user-images.githubusercontent.com/89016122/185281085-c12910d1-c234-4000-b923-4ef29573484e.png)

- the last classifying layer of the neural network is located in the object that calculates loss function;
- initial learning rates were set to 2e-4 for the feature extractor part and 1e-2 for the classifier weights;
- learning rate scheduler ReduceOnPlateau was used;
- heavy augmentations with Albumentation library were used;
- each epoch took approximately 45min of training and 5min of validation;
- first 8 epochs were trained with CCE loss, achieved val_accuracy of 0.96;
- next 22 epochs were trained with ArcFace loss, achieved val_accuracy of 0.92

|epoch| loss type | train_acc| val_acc|
|----|----|----|----|
|4  | CCE |0.79|0.88|
|8  | CCE |0.90|0.96|
|9  |ArcFace| 0.00|0.06|
|15 |ArcFace| 0.31|0.56|
|20 |ArcFace| 0.53|0.82|
|25 |ArcFace| 0.69|0.89|
|30 |ArcFace| 0.74|0.92|