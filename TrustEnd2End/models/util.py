import torch
import torch.nn.functional as F


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

    return torch.mean(A + B)


def sce_loss(p, alpha, c, smooth_factor = 0.6):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    label_s = label * smooth_factor + (1 - smooth_factor) / c
    A = torch.sum(label_s * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return torch.mean(A)


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
