import torch
import torch.nn.functional as F


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