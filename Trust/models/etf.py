import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import Fusion, ETrustNet


class ETF(nn.Module):
    
    def discount(self, func_evidence, refer_evidence):
        
        disc_evidence = dict()
        for v in range(len(func_evidence)):
            # func_evidence_v = func_evidence[v]
            # func_alpha_v = func_evidence_v + 1
            # func_s_v = torch.sum(func_alpha_v, dim=1, keepdim=True)
            # func_b_v = func_evidence_v / func_s_v.expand(func_evidence_v.shape)
            
            # refer_evidence_v = refer_evidence[v]
            # refer_alpha_v = refer_evidence_v + 1
            # refer_s_v = torch.sum(refer_alpha_v, dim=1, keepdim=True)
            # refer_p_v = refer_alpha_v[:, 1:2] / refer_s_v

            # disc_b_v = refer_p_v.expand(func_b_v.shape) * func_b_v
            # disc_u_v = 1 - torch.sum(disc_b_v, keepdim=True, dim=1)
            # disc_s_v = self.n_classes / disc_u_v
            
            # disc_evidence_v = torch.mul(disc_b_v, disc_s_v.expand(disc_b_v.shape))
            # disc_evidence[v] = disc_evidence_v
            
            func_evidence_v = func_evidence[v]
            func_alpha_v = func_evidence_v + 1
            func_s_v = torch.sum(func_alpha_v, dim=1, keepdim=True)
            func_u_v = func_evidence_v.shape[1] / func_s_v
            
            refer_evidence_v = refer_evidence[v]
            refer_alpha_v = refer_evidence_v + 1
            refer_s_v = torch.sum(refer_alpha_v, dim=1, keepdim=True)
            refer_p_v = refer_alpha_v[:, 1:2] / refer_s_v

            disc_evidence_v = refer_p_v * func_u_v / (1 - refer_p_v + refer_p_v * func_u_v) * func_evidence_v
            disc_evidence[v] = disc_evidence_v
            
        return disc_evidence

    def form_refer_feat(self, feat, func_evidence):
        
        refer_feat = dict()
        for v in range(len(func_evidence)):
            feat_v = feat[v]
            func_evidence_v = func_evidence[v]
            
            func_alpha_v = func_evidence_v + 1
            func_s_v = torch.sum(func_alpha_v, dim=1, keepdim=True)
            func_b_v = func_evidence_v / func_s_v.expand(func_evidence_v.shape)
            # func_u_v = self.n_classes / func_s_v
            
            if self.refer_bi:
                refer_feat_v = (feat_v.detach(), func_b_v.detach(),)
            else:  
                # F.one_hot(torch.argmax(func_b_v, 1), self.n_classes),
                # get_dc_loss(func_evidence, v, feat_v.device).detach(),
                # func_u_v.detach(),
                refer_feat_v = torch.cat([
                    feat_v.detach(),
                    func_b_v.detach(),
                    ], dim = 1)
                
            refer_feat[v] = refer_feat_v
            
        return refer_feat
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_views = args.n_views
        self.n_classes = args.n_classes
        self.views_dims = args.views_dims
        
        self.fuse = Fusion(args.n_classes, "bcf")
        # self.fuse = Fusion(args.n_classes, "abf")
        # self.fuse = Fusion(args.n_classes, "a-cbf")
        
        func_in_dims = args.views_dims.copy()
        func_in_dims.append(sum(args.views_dims))
        self.func_net = ETrustNet(args, func_in_dims, args.n_classes)
        
        self.refer_net = None
        self.refer_bi = True
        
    def init_refer_net(self):
        if self.refer_bi:
            refer_in_dims = [(self.args.views_dims[i], self.args.n_classes) for i in range(self.args.n_views)]
            refer_in_dims.append( (sum(self.args.views_dims), self.args.n_classes) )
        else:
            refer_in_dims = [self.args.views_dims[i]+self.args.n_classes for i in range(self.args.n_views)]
            refer_in_dims.append(sum(self.args.views_dims)+self.args.n_classes)
        self.refer_net = ETrustNet(self.args, refer_in_dims, 2, bi=self.refer_bi)
        
    def forward(self, X):
        func_feat = dict()
        all_feat = []
        for v_num in range(self.n_views):
            func_feat[v_num] = X[v_num]
            all_feat.append(X[v_num])
        func_feat[self.n_views] = torch.cat(all_feat, dim=1)
        func_evidence = self.func_net(func_feat)
        func_alpha = {k: v+1 for k, v in func_evidence.items()}
        
        if self.refer_net is not None:
            refer_feat = self.form_refer_feat(func_feat, func_evidence)
            refer_evidence = self.refer_net(refer_feat)
            refer_alpha = {k: v+1 for k, v in refer_evidence.items()}

            disc_evidence = self.discount(func_evidence, refer_evidence)
            disc_alpha = {v_num: evidence_v + 1 for v_num, evidence_v in disc_evidence.items()}
            
            evidence_a = self.fuse(disc_evidence)
        else:
            refer_alpha = None
            disc_alpha = None
            evidence_a = self.fuse(func_evidence)
        
        alpha_a = evidence_a + 1
        
        return func_alpha, refer_alpha, disc_alpha, alpha_a
