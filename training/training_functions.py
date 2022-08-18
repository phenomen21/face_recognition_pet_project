import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F
from tqdm import tqdm, tqdm_notebook
from ...utils.losses import StandCrossEntropyLoss, ArcFaceLoss, cosine_distance
from ...utils.datasets import MyCompose, MyRandomHorizontalFlip, celebADataset, celebADatasetTriplet, init_triplet_loaders



def fit_epoch(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for batch in tqdm(train_loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
        inputs = batch['images'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
        del inputs
        del labels
        torch.cuda.empty_cache()
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc



def eval_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    print('Validating model:')
    for batch in tqdm(val_loader):
        inputs = batch['images'].to(device)
        labels = batch['labels'].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
        del inputs
        del labels
        torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    val_acc = running_corrects.cpu().numpy() / processed_size
    return val_loss, val_acc



def train(train_loader, val_loader, model, criterion, epochs, opt, sched=None, start_epoch=0, model_name='', device='cpu'):
    history = []
    log_template = "Epoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    for epoch in range(epochs):
      print('\nEpoch {:03d}/{:03d} is going: LR_class = {:.2e}, LR_feat={:.2e}'.format(epoch+1+start_epoch, epochs+start_epoch,
                                                                                           opt.param_groups[0]['lr'], opt.param_groups[1]['lr']))
      train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt, device)

      val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
      history.append((train_loss, train_acc, val_loss, val_acc))
      if sched is not None:
        sched.step(val_acc)
      print(log_template.format(ep=epoch+1+start_epoch, t_loss=train_loss,\
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
      model_filename = '{}_epoch_{:02d}'.format(model_name, epoch+1+start_epoch)
      torch.save(model, model_filename)
    return history



def fit_epoch_triplet(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
    batch_n=0
    mean_emb = dict()
    model.train()
    for batch in tqdm(train_loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
        inputs = batch['images'].to(device)
        labels = batch['labels'].detach().cpu().numpy()
        inputs_pos = batch['images_pos'].to(device)
        inputs_neg = batch['images_neg'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs_pos = model(inputs_pos)
        outputs_neg = model(inputs_neg)

        # to measure accuracy we calculate "mean" embeddings for every class during training
        for i, lab in enumerate(labels):
          vec = outputs[i].cpu().detach()
          if lab in mean_emb:
            mean_emb[lab]['avg_emb'] = (mean_emb[lab]['avg_emb'] * mean_emb[lab]['count'] + vec) / (mean_emb[lab]['count']+1)
            mean_emb[lab]['count'] += 1
          else:
            mean_emb[lab] = {'count':1,'avg_emb':vec}
        loss = criterion(outputs, outputs_pos, outputs_neg)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)
        del inputs
        del labels
        del inputs_pos
        del inputs_neg
        torch.cuda.empty_cache()
    train_loss = running_loss / processed_data
    return train_loss, mean_emb



def eval_epoch_triplet(model, val_loader, criterion, mean_embeddings, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    cos_sims = 0
    val_cnt = 0
    print('Validating model:')
    for batch in tqdm(val_loader):
      inputs = batch['images'].to(device)
      labels = batch['labels'].detach().cpu().numpy()
      inputs_pos = batch['images_pos'].to(device)
      inputs_neg = batch['images_neg'].to(device)

      with torch.set_grad_enabled(False):
        outputs = model(inputs)
        outputs_pos = model(inputs_pos)
        outputs_neg = model(inputs_neg)
        loss = criterion(outputs, outputs_pos, outputs_neg)
      for i, lab in enumerate(labels):
        outp = outputs[i].detach().cpu()
        cossim = cosine_distance(outp,mean_embeddings[lab]['avg_emb'])
        cos_sims += cossim
      val_cnt += inputs.size(0)
      running_loss += loss.item() * inputs.size(0)

      processed_size += inputs.size(0)
      del inputs
      del labels
      del inputs_pos
      del inputs_neg
      torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    cos_sims /= val_cnt
    return val_loss, cos_sims



def train_triplet(train_loader, val_loader, model, criterion, epochs, opt, sched=None, start_epoch=0, device='cpu', model_name=''):
    history = []
    global val_cossim_pred
    log_template = "Epoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} val_cos_dist {v_cos:0.4f} pred_val {pred_v:0.4f}"

    for epoch in range(epochs):
      print('\nEpoch {:03d}/{:03d} is going: LR = {:.2e}'.format(epoch+1+start_epoch, epochs+start_epoch,
                                                                                           opt.param_groups[0]['lr']))
      train_loss, mean_emb = fit_epoch_triplet(model, train_loader, criterion, opt, device=device)

      # Save model before validation (25 min on full dataset)
      model_filename = '{}_triplet_epoch_{:02d}'.format(model_name, epoch+1+start_epoch)
      torch.save(model, model_filename)

      val_loss, val_cossim = eval_epoch_triplet(model, val_loader, criterion, mean_embeddings=mean_emb, device=device)
      history.append((train_loss, val_loss, val_cossim))
      if sched is not None:
        sched.step(val_cossim)
      print(log_template.format(ep=epoch+1+start_epoch, t_loss=train_loss, v_loss=val_loss, v_cos=val_cossim, pred_v=val_cossim_pred))
      if val_cossim > val_cossim_pred:
        opt.param_groups[0]['lr'] /= 2
      val_cossim_pred = val_cossim
    return history



def fit_epoch_arc(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
    batch_n=0
    model.train()
    for batch in tqdm(train_loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
        inputs = batch['images'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels, loss=True)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(criterion(outputs, labels, loss=False), 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
        del inputs
        del labels
        torch.cuda.empty_cache()
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc



def eval_epoch_arc(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    print('Validating model:')
    for batch in tqdm(val_loader):
        inputs = batch['images'].to(device)
        labels = batch['labels'].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels, loss=True)
            preds = torch.argmax(criterion(outputs, labels, loss=False), 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
        del inputs
        del labels
        torch.cuda.empty_cache()
    val_loss = running_loss / processed_size
    val_acc = running_corrects.cpu().numpy() / processed_size
    return val_loss, val_acc



def train_arc(train_loader, val_loader, model, criterion, epochs, opt, sched=None, start_epoch=0, device='cpu', model_name=''):
    history = []
    log_template = "Epoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    for epoch in range(epochs):
      print('\nEpoch {:03d}/{:03d} is going: LR_class = {:.2e}, LR_feat={:.2e}'.format(epoch+1+start_epoch, epochs+start_epoch,
                                                                                           opt.param_groups[0]['lr'], opt.param_groups[1]['lr']))
      train_loss, train_acc = fit_epoch_arc(model, train_loader, criterion, opt, device=device)

      val_loss, val_acc = eval_epoch_arc(model, val_loader, criterion, device)
      history.append((train_loss, train_acc, val_loss, val_acc))
      if sched is not None:
        sched.step(val_acc)
      print(log_template.format(ep=epoch+1+start_epoch, t_loss=train_loss,\
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
      model_filename = '{}_arc_epoch_{:02d}'.format(model_name, epoch+1+start_epoch)
      torch.save(model, model_filename)
      loss_filename = model_filename + '_loss'
      torch.save(criterion, loss_filename)
    return history
