import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, tqdm_notebook
from datasets import MyCompose, MyRandomHorizontalFlip


def get_embeddings_from_loader(dataloader, model, max_emb=5000):
  emb = None
  emb_num = 0
  for batch in tqdm(dataloader):
    images = batch['images'].to(DEVICE)
    batch_emb_num = images.shape[0]
    if emb_num + batch_emb_num > max_emb:
      images = images[0:max_emb-emb_num,:,:,:]
    emb_num += images.shape[0]
    batch_emb = model(images)
    batch_emb = batch_emb.cpu().detach().numpy()
    if emb is None:
      emb = batch_emb
    else:
      emb = np.vstack((emb,batch_emb))
    del images
    del batch
    torch.cuda.empty_cache()
    if emb_num >= max_emb:
      break
  return emb


def get_embeddings_from_dataframes(X, Y, model, max_emb=5000):
  '''
  This function builds dataset and dataloader from dataframes and returns embeddings from it
  '''
  transform = MyCompose([
      MyRandomHorizontalFlip(0),
      tt.Resize((RESCALE_SIZE, RESCALE_SIZE)),
      tt.Normalize(*stats)])
  temp_dataset = celebADataset(X, Y, transform)
  temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=32, shuffle=False, num_workers=2)
  embeddings = get_embeddings_from_loader(temp_loader, model, max_emb)
  return embeddings

