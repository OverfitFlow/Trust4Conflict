import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet152, resnet50
from .util import Fusion


class ECML(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(args.lm_name)
        self.img_encoder = resnet50(pretrained=True)
        
        self.text_clf = nn.Sequential(*[
            nn.GELU(),
            nn.Linear(768, args.n_classes)
        ])
        
        self.img_clf = nn.Sequential(*[
            nn.ReLU(),
            nn.Linear(1000, args.n_classes)
        ])
        
        self.fuser = Fusion(args.n_classes, 'abf')
        
    def forward(self, i, t):
        
        ht = self.text_encoder(**t).pooler_output
        hi = self.img_encoder(i)
        t_evi = F.softplus(self.text_clf(ht))
        i_evi = F.softplus(self.img_clf(hi))
        
        i_t_evi = self.fuser.fuse_two(i_evi, t_evi)
        
        return (
            i_evi, t_evi, i_t_evi
        )
        

def get_dc_loss(evidences):
    device = evidences[0].device
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum
