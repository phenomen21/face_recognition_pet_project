import torch
from ..training_functions import StandCrossEntropyLoss, ArcFaceLoss, train_land
from ...utils.datasets import init_loaders


import timm

stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
RESCALE_SIZE = 200

# suppose that dataset is already downloaded and located in 'celeba_dataset' directory - I intentionally left it empty
# bboxes and landmarks should already be in the dataset folder - files 'list_bbox_celeba.txt', 'list_landmarks_celeba.txt' and 'identity_celeba.txt'

if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device("cuda")

model_name = 'rexnet_200_landmarks'
train_loader, val_loader, test_loader, n_classes = init_loaders('../celeba_dataset', batch_size=32, prob=0.20, rescale_size=RESCALE_SIZE,align=False)

# create a model with 10 output neurons
model_arc = timm.create_model(model_name, pretrained=True, num_classes=10).to(DEVICE)

# MSE loss for the regression task
cce_loss = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(params=[
        {"params": cce_loss.parameters(), "lr":1e-2},
        {"params": model_arc.features.parameters(), "lr":2e-4}], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, factor=0.5, threshold_mode='rel',cooldown=1,threshold=0.05)

# train
history = train_land(train_loader, val_loader, model=model_arc, criterion=cce_loss, epochs=5, opt=optimizer, sched=scheduler,start_epoch=0, 
                                                                                                device=DEVICE, model_name=model_name)
