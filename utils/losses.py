import torch
import torch.nn as nn



class StandCrossEntropyLoss(nn.Module):
    '''
    Standard CCE loss from embeddings, includes trainable fully-connected layer
    '''
    def __init__(self, in_features, out_features, eps=1e-7, params=None):
        super(StandCrossEntropyLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=True)
        if params is not None:
          self.fc.weight = torch.nn.Parameter(params, requires_grad=True)
        self.eps = eps

    def forward(self, x, labels, loss=True):
        wf = self.fc(x)
        if loss:
          return torch.nn.functional.cross_entropy(wf,labels,reduction='mean')
        else:
          return wf


class ArcFaceLoss(nn.Module):
  '''
  ArcFaceLoss as described here - https://arxiv.org/pdf/1801.07698.pdf
  when loss=True returns CCE loss from logits, when loss=False returns logits themselves
  '''

  def __init__(self, emb_features, num_classes, s=30.0, m=0.50, eps=1e-8, params=None):
    super(ArcFaceLoss, self).__init__()
    self.emb_features = emb_features
    self.num_classes = num_classes
    self.s = s
    self.m = m
    self.eps = eps
    if params is not None:
        self.weight = nn.Parameter(params)
    else:
        self.weight = nn.Parameter(torch.randn(emb_features, num_classes))

  def forward(self, inputs, labels, loss=True):
    weights = torch.nn.functional.normalize(self.weight, p=2, dim=1)
    inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)
    cos_theta = torch.nn.functional.linear(inputs, weights)
    arccos_theta = torch.acos(torch.clamp(cos_theta,self.eps,1-self.eps))
    m_hot = torch.tensor(labels.detach().clone()).unsqueeze(0)
    m_hot = torch.zeros(m_hot.size(0), n_classes).to(DEVICE).scatter_(1, m_hot.detach().clone(), 1.)
    cos_theta_m = torch.cos(arccos_theta + self.m * m_hot)
    logits_new = cos_theta_m * self.s
    if loss:
      return torch.nn.functional.cross_entropy(logits_new, labels)
    else:
      return logits_new
