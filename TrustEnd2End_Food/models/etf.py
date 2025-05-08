import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet152, resnet50
from .util import Fusion


class ETF(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(args.lm_name)
        self.img_encoder = resnet50(pretrained=True)
        
        self.text_clf = nn.Linear(768, args.n_classes)
        self.img_clf = nn.Linear(1000, args.n_classes)
        self.fuser = Fusion(args.n_classes, 'bcf')
        
        self.it_clf = nn.Linear(1000+768, args.n_classes)
        
        self.refer_text = nn.Bilinear(768, args.n_classes, 2)
        self.refer_img = nn.Bilinear(1000, args.n_classes, 2)
        self.refer_it = nn.Bilinear(1000+768, args.n_classes, 2)
        
    def forward(self, i, t):
        
        ht = F.gelu(self.text_encoder(**t).pooler_output)
        hi = F.relu(self.img_encoder(i))
        t_evi = F.softplus(self.text_clf(ht))
        i_evi = F.softplus(self.img_clf(hi))

        hit = torch.cat([hi, ht], dim=-1)
        it_evi = F.softplus(self.it_clf(hit))
        
        i_refer_feat = self.form_refer_feat(hi, i_evi)
        i_refer_evi = F.softplus(self.refer_img(i_refer_feat[0], i_refer_feat[1]))
        t_refer_feat = self.form_refer_feat(ht, t_evi)
        t_refer_evi = F.softplus(self.refer_text(t_refer_feat[0], t_refer_feat[1]))
        
        it_refer_feat = self.form_refer_feat(hit, it_evi)
        it_refer_evi = F.softplus(self.refer_it(it_refer_feat[0], it_refer_feat[1]))
        
        i_disc_evi = self.discount(i_evi, i_refer_evi)
        t_disc_evi = self.discount(t_evi, t_refer_evi)
        it_disc_evi = self.discount(it_evi, it_refer_evi)
        
        i_t_evi = self.fuser.fuse_two(self.fuser.fuse_two(i_disc_evi, t_disc_evi), it_disc_evi)
        
        return (
            i_evi, t_evi, it_evi,
            i_refer_evi, t_refer_evi, it_refer_evi,
            i_disc_evi, t_disc_evi, it_disc_evi,
            i_t_evi
        )

    def form_refer_feat(self, feat_v, func_evidence_v):
    
        func_alpha_v = func_evidence_v + 1
        func_s_v = torch.sum(func_alpha_v, dim=1, keepdim=True)
        func_b_v = func_evidence_v / func_s_v.expand(func_evidence_v.shape)
        refer_feat_v = (feat_v.detach(), func_b_v.detach(),)
            
        return refer_feat_v
    
    def discount(self, func_evidence_v, refer_evidence_v):
            
        func_alpha_v = func_evidence_v + 1
        func_s_v = torch.sum(func_alpha_v, dim=1, keepdim=True)
        func_u_v = func_evidence_v.shape[1] / func_s_v
        
        refer_alpha_v = refer_evidence_v + 1
        refer_s_v = torch.sum(refer_alpha_v, dim=1, keepdim=True)
        refer_p_v = refer_alpha_v[:, 1:2] / refer_s_v

        disc_evidence_v = refer_p_v * func_u_v / (1 - refer_p_v + refer_p_v * func_u_v) * func_evidence_v
            
        return disc_evidence_v
    

def sce_loss(p, alpha, c, smooth_factor = 0.6):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    label_s = label * smooth_factor + (1 - smooth_factor) / c
    A = torch.sum(label_s * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return A