import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(classifier_dims, classes),
            nn.Softplus()
        )
        
    def forward(self, x):
        h = self.fc(x)
        return h
    

class Feater(nn.Module):
    def __init__(self, in_dims, out_dim):
        super().__init__()
        self.fc = nn.Bilinear(in_dims[0], in_dims[1], out_dim)
        self.act = nn.Softplus()
        
    def forward(self, x1, x2):
        h = self.act(self.fc(x1, x2))
        return h


# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


def sce_loss(p, alpha, c, smooth_factor = 0.6):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    label_s = label * smooth_factor + (1 - smooth_factor) / c
    A = torch.sum(label_s * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return A


def get_dc_loss(evidences, device):
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


class Fusion:
    
    def __init__(self, n_classes, fuse_method="bcf"):
        self.n_classes = n_classes
        self.fuse_method = fuse_method
        if self.fuse_method == "bcf":
            self.fuse_two = self.constraint_fuse
        elif self.fuse_method == "wbf":
            self.fuse_two = self.weighted_avg_fuse
        elif self.fuse_method == 'a-cbf':
            self.fuse_two = self.cumm_fuse
        elif self.fuse_method == "abf":
            self.fuse_two = self.avg_fuse
        else:
            raise NotImplementedError
    
    def constraint_fuse2(self, evidence1, evidence2):
        # Calculate the merger of two DS evidences
        E = dict()
        E[0], E[1] = evidence1, evidence2
        b, S, alpha, u = dict(), dict(), dict(), dict()
        for v in range(2):
            alpha[v] = E[v] + 1
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            b[v] = E[v] / S[v].expand(E[v].shape)
            u[v] = self.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.n_classes, 1), b[1].view(-1, 1, self.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        return e_a
    
    def constraint_fuse(self, evidence1, evidence2):
        return evidence1 + evidence2 + (evidence1 * evidence2) / evidence1.shape[1]
    
    def weighted_avg_fuse(self, evidence1, evidence2):
        alpha1 = evidence1 + 1
        u1 = self.n_classes / torch.sum(alpha1, dim=1, keepdim=True)
        alpha2 = evidence2 + 1
        u2 = self.n_classes / torch.sum(alpha2, dim=1, keepdim=True)
        e_a = ((1-u1) * evidence1 + (1-u2) * evidence2) / (2 - u1 - u2)
        return e_a
    
    def avg_fuse(self, evidence1, evidence2):
        return (evidence1 + evidence2) / 2
    
    def cumm_fuse(self, evidence1, evidence2):
        return evidence1 + evidence2
    
    def fuse(self, evidence):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        for v in range(len(evidence)-1):
            if v == 0:
                evidence_a = self.fuse_two(evidence[0], evidence[1])
            else:
                evidence_a = self.fuse_two(evidence_a, evidence[v+1])
        return evidence_a
    
    def __call__(self, *args, **kwargs):
        return self.fuse(*args, **kwargs)
    
    
class ETrustNet(nn.Module):
    
    def __init__(self, args, in_dims, out_dim, bi=False):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super().__init__()
        self.args = args
        self.n_views = args.n_views # raw #views
        self.n_classes = args.n_classes
        assert len(in_dims) == args.n_views + 1
        
        self.bi = bi
        if bi:
            self.clfs = nn.ModuleList([Feater(in_dims[i], out_dim) for i in range(args.n_views)])
            self.clf_ps = Feater(in_dims[-1], out_dim)
        else:
            self.clfs = nn.ModuleList([Classifier(in_dims[i], out_dim) for i in range(args.n_views)])
            self.clf_ps = Classifier(in_dims[-1], out_dim)
    
    def forward(self, inp):
        evidence = dict()
        for v in range(self.n_views):
            if self.bi:
                evidence[v] = self.clfs[v](inp[v][0], inp[v][1])
            else:
                evidence[v] = self.clfs[v](inp[v])
        if self.bi:
            evidence[self.n_views] = self.clf_ps(inp[self.n_views][0], inp[self.n_views][1])
        else:
            evidence[self.n_views] = self.clf_ps(inp[self.n_views])
        return evidence
    

class TrustNet(nn.Module):
    
    def __init__(self, args, in_dims, out_dim, bi=False):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super().__init__()
        self.args = args
        self.n_views = args.n_views # raw #views
        self.n_classes = args.n_classes
        assert len(in_dims) == args.n_views
        
        self.bi = bi
        if bi:
            self.clfs = nn.ModuleList([Feater(in_dims[i], out_dim) for i in range(args.n_views)])
        else:
            self.clfs = nn.ModuleList([Classifier(in_dims[i], out_dim) for i in range(args.n_views)])
    
    def forward(self, inp):
        evidence = dict()
        for v in range(self.n_views):
            if self.bi:
                evidence[v] = self.clfs[v](inp[v][0], inp[v][1])
            else:
                evidence[v] = self.clfs[v](inp[v])
        return evidence