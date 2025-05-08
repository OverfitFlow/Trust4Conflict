import torch
import torch.nn as nn
import torch.nn.functional as F
from models.image import ImageEncoder
from models.util import Fusion


class ETF(nn.Module):
    
    def form_refer_feat(self, feat_v, func_evidence_v):
    
        func_alpha_v = func_evidence_v + 1
        func_s_v = torch.sum(func_alpha_v, dim=1, keepdim=True)
        func_b_v = func_evidence_v / func_s_v.expand(func_evidence_v.shape)
        refer_feat_v = (feat_v, func_b_v.detach(),)
            
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
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fuser = Fusion(args.n_classes, args.belief_fusion)
        
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        self.clf_depth = nn.Linear(depth_last_size, args.n_classes)
        self.clf_rgb = nn.Linear(rgb_last_size, args.n_classes)
        
        self.refer_depth = nn.Bilinear(depth_last_size, args.n_classes, 2)
        self.refer_rgb = nn.Bilinear(rgb_last_size, args.n_classes, 2)
        
        self.clf_ps = nn.Linear(depth_last_size+rgb_last_size, args.n_classes)
        self.refer_ps = nn.Bilinear(depth_last_size + rgb_last_size, args.n_classes, 2)
        
    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)

        depth_func_out = self.clf_depth(depth)
        rgb_func_out = self.clf_rgb(rgb)
        depth_func_evidence, rgb_func_evidence = F.softplus(depth_func_out), F.softplus(rgb_func_out)
        depth_func_alpha, rgb_func_alpha = depth_func_evidence+1, rgb_func_evidence+1
        
        depth_refer_feat = self.form_refer_feat(depth, depth_func_evidence)
        depth_refer_evidence = F.softplus(self.refer_depth(depth_refer_feat[0], depth_refer_feat[1]))
        rgb_refer_feat = self.form_refer_feat(rgb, rgb_func_evidence)
        rgb_refer_evidence = F.softplus(self.refer_rgb(rgb_refer_feat[0], rgb_refer_feat[1]))
        
        ps = torch.cat([depth, rgb], dim=1)
        ps_func_out = self.clf_ps(ps)
        ps_func_evidence = F.softplus(ps_func_out)
        ps_func_alpha = ps_func_evidence + 1
        ps_refer_feat = self.form_refer_feat(ps, ps_func_evidence)
        ps_refer_evidence = F.softplus(self.refer_ps(ps_refer_feat[0], ps_refer_feat[1]))
        
        depth_disc_evidence = self.discount(depth_func_evidence, depth_refer_evidence)
        rgb_disc_evidence = self.discount(rgb_func_evidence, rgb_refer_evidence)
        depth_rgb_evidence = self.fuser.fuse_two(depth_disc_evidence, rgb_disc_evidence)
        ps_disc_evidence = self.discount(ps_func_evidence, ps_refer_evidence)
        depth_rgb_ps_evidence = self.fuser.fuse_two(depth_rgb_evidence, ps_disc_evidence)
        
        depth_rgb_alpha = depth_rgb_ps_evidence + 1
        
        return (
            depth_func_alpha, rgb_func_alpha, ps_func_alpha,
            depth_refer_evidence+1, rgb_refer_evidence+1, ps_refer_evidence+1,
            depth_disc_evidence+1, rgb_disc_evidence+1, ps_disc_evidence+1,
            depth_rgb_alpha
        )
        