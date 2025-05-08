import logging
import argparse
from models.util import ce_loss, sce_loss, get_dc_loss
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.inter_rater import fleiss_kappa


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def get_logger(args):
    logger = logging.getLogger("Logger")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(f"logs/{args.log_file}")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    
    parser.add_argument('--annealing-step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--rlr', type=float, default=0.0003, metavar='RLR',
                        help='learning rate for referral trust net')
    
    parser.add_argument('--data-name', type=str, default="Scene",
                        help='')
    parser.add_argument('--data-path', type=str, default="datasets/ICLR2021 Datasets",
                        help='')
    
    parser.add_argument('--log-file', type=str, default="out",
                        help='log file path')
    
    parser.add_argument("--model-name", type=str, default="ETF")
    parser.add_argument("--conflict-test", action='store_true')
    parser.add_argument("--conflict-sigma", type=float)
    
    parser.add_argument("--smooth-factor", type=float, default=0.6)
    parser.add_argument("--warm-epochs", type=int, default=5)
    parser.add_argument("--used-views", nargs="*", default=[-1])
    args = parser.parse_args()
    
    return args



def train_func(args, dl, model, optimizer_func, optimizer_refer, optimizer_joint, epoch):
    model.train()
    for X, y in dl:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].cuda()
        y = y.long().cuda()
        
        # 1 分开train
        func_alpha, *_ = model(X)
        optimizer_func.zero_grad() #, optimizer_refer.zero_grad()
        loss = 0
        for v_num in range(len(func_alpha)):
            loss += ce_loss(y, func_alpha[v_num], args.n_classes, epoch, args.annealing_step)
        loss.sum().backward()
        optimizer_func.step() #, optimizer_refer.step()
        
        # 2 一起train discount
        *_, alpha_a = model(X)
        optimizer_func.zero_grad()
        loss = ce_loss(y, alpha_a, args.n_classes, epoch, args.annealing_step)
        loss.sum().backward()
        optimizer_func.step()


def train_warmup(args, dl, model, optimizer_func, optimizer_refer, optimizer_joint, epoch):
    model.train()
    for X, y in dl:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].cuda()
        y = y.long().cuda()

        func_alpha, refer_alpha, *_ = model(X)
        
        optimizer_refer.zero_grad()
        loss = 0
        for v_num in range(len(func_alpha)):
            t_v = (torch.argmax(func_alpha[v_num], 1) == y).to(torch.long)
            loss += sce_loss(t_v, refer_alpha[v_num], 2, args.smooth_factor)
        loss.sum().backward()
        optimizer_refer.step()
             

def train_refer(args, dl, model, optimizer_func, optimizer_refer, optimizer_joint, epoch):
    model.train()
    for X, y in dl:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].cuda()
        y = y.long().cuda()

        *_, alpha_a = model(X)
        optimizer_refer.zero_grad()#, optimizer_func.zero_grad()
        loss = ce_loss(y, alpha_a, args.n_classes, epoch, args.annealing_step)
        loss.sum().backward()
        optimizer_refer.step()#, optimizer_func.step()


def test(args, dl, model, epoch):
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    
    model.eval()
    for X, y in dl:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].cuda()
        y = y.long().cuda()
        
        func_alpha, *_, alpha_a = model(X)
        
        loss = ce_loss(y, alpha_a, args.n_classes, epoch, args.annealing_step)
        for v_num in range(len(func_alpha)):
            loss += ce_loss(y, func_alpha[v_num], args.n_classes, epoch, args.annealing_step)
        
        loss_meter.update(loss.mean().detach().cpu().item(), y.shape[0])
        corr = (torch.argmax(alpha_a, 1) == y).to(torch.long).sum().cpu().item()
        acc_meter.update(corr / y.shape[0], y.shape[0])

    return loss_meter.avg, acc_meter.avg


def test_final(args, dl, model, epoch):
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    
    labels = []
    preds = []
    uncer = []
    
    views_preds = {}
    
    # criteria 1
    # if at least one view is corret
    # how many joint are wrong (less is better)
    one_view_corr_count = 0
    # one_view_corr_joint_corr_count = 0
    n_instance = 0

    model.eval()
    for X, y in dl:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].cuda()
        y = y.long().cuda()
        
        with torch.no_grad():
            *_, disc_alpha, alpha_a = model(X)
        loss = ce_loss(y, alpha_a, args.n_classes, epoch, args.annealing_step)
        for v_num in range(len(disc_alpha)):
            loss += ce_loss(y, disc_alpha[v_num], args.n_classes, epoch, args.annealing_step)
            if v_num not in views_preds:
                views_preds[v_num] = []
            views_preds[v_num].extend(torch.argmax(disc_alpha[v_num], dim=1).cpu().numpy().tolist())

        loss_meter.update(loss.mean().detach().cpu().item(), y.shape[0])
        corr = (torch.argmax(alpha_a, 1) == y).to(torch.long).sum().cpu().item()
        acc_meter.update(corr / y.shape[0], y.shape[0])
        
        u = args.n_classes / torch.sum(alpha_a, dim=1)
        uncer.extend(u.cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())
        preds_ = torch.argmax(alpha_a, 1)
        # mask_ = (alpha_a == alpha_a.max(dim=1, keepdim=True)[0]).sum(dim=1) > 1
        # preds_[mask_] = -1
        preds.extend(preds_.cpu().numpy().tolist())
        
        each_view_corr_mask = torch.stack([torch.argmax(v, dim=1) == y for v in disc_alpha.values()], dim=1)  # B x V: bool
        # one_view_corr_clip = torch.clip(each_view_corr_mask.sum(dim=-1), max=1)  # c1
        # one_view_corr_clip = (each_view_corr_mask.sum(dim=-1) == 1).to(torch.float)  # c2
        # one_view_corr_clip = (each_view_corr_mask.sum(dim=-1) > 1).to(torch.float)
        one_view_corr_clip = (each_view_corr_mask.sum(dim=-1) > (args.n_views // 2)).to(torch.float)
        one_view_corr_count += one_view_corr_clip.sum().cpu().item()
        # one_view_corr_mask = one_view_corr_clip.to(torch.bool)
        # one_view_corr_joint_corr_count += (torch.argmax(alpha_a, dim=1) == y)[one_view_corr_mask].sum().cpu().item()
        n_instance += y.shape[0]
    
    views_preds = np.vstack([value for value in views_preds.values()]).T
    n_data, n_views = views_preds.shape
    counts = np.zeros((n_data, args.n_classes), dtype=int)
    
    for i in range(n_data):
        for j in range(n_views):
            counts[i, views_preds[i, j]] += 1
    
    correctness = (np.array(preds) != np.array(labels)).astype(int)
    conf = np.array(uncer)  # high uncer for 1, low for 0, 0 is correct prediction
    if len(np.unique(correctness)) == 1:  # handle edge case
        artifact_value = 1 - np.unique(correctness)
        correctness = np.append(correctness, int(artifact_value))
        conf = np.append(conf, float(artifact_value))
    conf_auc_roc = roc_auc_score(correctness, conf)
    
    return (
        loss_meter.avg,
        acc_meter.avg,
        conf_auc_roc,
        fleiss_kappa(counts),
        # one_view_corr_joint_corr_count/one_view_corr_count,
        one_view_corr_count / n_instance
        )
