import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet152, resnet50
from .util import Fusion


class CCML(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
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
        
    def forward(self, i, t, beta=1.25):
        ht = self.text_encoder(**t).pooler_output
        hi = self.img_encoder(i)
        t_evi = F.softplus(self.text_clf(ht))
        i_evi = F.softplus(self.img_clf(hi))

        func_alpha = {0: i_evi+1, 1: t_evi+1}
        
        alpha_a, alpha_con, alpha_div = self.Evidence_DC(func_alpha, beta)
        evidence_a = alpha_a - 1
        evidence_con = alpha_con - 1
        evidence_div = alpha_div - 1
        evidences = {0: i_evi, 1: t_evi}
        
        return evidences, evidence_a, evidence_con, evidence_div

    def Evidence_DC(self, alpha, beta):
        E = dict()
        for v in range(len(alpha)):
            E[v] = alpha[v]-1
            E[v] = torch.nan_to_num(E[v], 0)

        for v in range(len(alpha)):
            E[v] = torch.nan_to_num(E[v], 0)

        E_con = E[0]
        for v in range(1, len(alpha)):
            E_con = torch.min(E_con, E[v])
        for v in range(len(alpha)):
            E[v] = torch.sub(E[v], E_con)
        alpha_con = E_con + 1

        E_div = E[0]
        for v in range(1,len(alpha)):
            E_div = torch.add(E_div, E[v])

        E_div = torch.div(E_div, len(alpha))

        S_con = torch.sum(alpha_con, dim=1, keepdim=True)

        b_con = torch.div(E_con, S_con)
        S_b = torch.sum(b_con, dim=1, keepdim=True)

        b_con2 = torch.pow(b_con, beta)
        S_b2 = torch.sum(b_con2,dim=1, keepdim=True)

        b_cona = torch.mul(b_con2, torch.div(S_b, S_b2))

        E_con = torch.mul(b_cona, S_con)

        E_con = torch.mul(E_con, len(alpha))
        E_a = torch.add(E_con, E_div)

        alpha_a = E_a + 1
        alpha_con = E_con + 1

        alpha_a = torch.nan_to_num(alpha_a, 0)
        alpha_con = torch.nan_to_num(alpha_con, 0)
        alpha_div = torch.nan_to_num(E_div+1, 0)

        Sum = torch.sum(alpha_a, dim=1, keepdim=True)
        return alpha_a, alpha_con, alpha_div
    

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def kl_loss(alpha, y, epoch_num, num_classes, annealing_step, device):
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    # print(kl_div)
    return torch.mean(kl_div)

def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    if not useKL:
        return loglikelihood

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=False):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return torch.mean(loss)


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def sd(evidences, device):
    alpha = evidences + 1
    S = torch.sum(alpha, dim=1, keepdim=True).to(device)
    b = evidences/(S.expand(evidences.shape)).to(device)
    batch_size, classes =evidences.shape[0], evidences.shape[1]
    SD = torch.zeros((batch_size)).to(device)

    b = b.t()
    for i in range(classes):
        for j in range(classes):
            if(i!=j):
                new = 1 - torch.abs(torch.sub(b[i], b[j]))
                SD = torch.add(SD, new.t())

    ans = torch.mean(SD)
    ans = torch.where(torch.isnan(ans), torch.zeros_like(ans), ans)

    return ans


def diss(evidences, device):
    alpha = evidences + 1
    S = torch.sum(alpha, dim=1, keepdim=True).to(device)
    b = evidences/(S.expand(evidences.shape)).to(device)
    b = torch.where(torch.isnan(b), torch.zeros_like(b), b)
    batch_size, classes =evidences.shape[0], evidences.shape[1]
    diss = torch.zeros((batch_size, classes)).to(device)

    b = b.t()
    for k in range(classes):
        denominator = torch.clip(torch.sum(b, dim=0, keepdim=True) - b[k], min=1e-8)
        oth = b[k] / denominator
        numerator = torch.abs(b - b[k])
        denominator = b + b[k]
        denominator = torch.clip(denominator, min=1e-8)
        Bal = 1 - torch.nan_to_num(torch.div(numerator, denominator), 0)
        Bal = Bal.masked_fill(Bal == 1, 0)

        diss += (b * oth * Bal).t()

    ans = torch.mean(diss)
    ans = torch.where(torch.isnan(ans), torch.zeros_like(ans), ans)
    print(ans)

    return ans

def get_loss(evidences, evidence_a, evidence_con, evidence_div, target, epoch_num, num_classes, annealing_step, gamma, delta, device):
    target = F.one_hot(target, num_classes)
    alpha_con = evidence_con + 1
    alpha_div = evidence_div + 1
    loss_acc = 0
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
        loss_acc += kl_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += edl_digamma_loss(alpha_con, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += gamma * edl_digamma_loss(alpha_div, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += delta * kl_loss(alpha_con, target, epoch_num, num_classes, annealing_step, device)

    loss = loss_acc
    if loss.isnan():
        print(loss)

    return loss