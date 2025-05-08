import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet152, resnet50
from .util import Fusion


class ETMC(nn.Module):
    
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
        
        self.fuser = Fusion(args.n_classes, 'bcf')
        
        self.it_clf = nn.Sequential(
            nn.Linear(1000+768, args.n_classes)
        )
        
    def forward(self, i, t):
        
        ht = self.text_encoder(**t).pooler_output
        hi = self.img_encoder(i)
        t_out = self.text_clf(ht)
        i_out = self.img_clf(hi)
        t_evi = F.softplus(t_out)
        i_evi = F.softplus(i_out)

        hit = torch.cat([F.relu(hi), F.gelu(ht)], dim=-1)
        it_evi = F.softplus(self.it_clf(hit))
        
        i_t_evi_ = self.fuser.fuse_two(i_evi, t_evi)
        i_t_evi = self.fuser.fuse_two(i_t_evi_, it_evi)
        
        return (
            i_evi, t_evi, it_evi, i_t_evi
        )
