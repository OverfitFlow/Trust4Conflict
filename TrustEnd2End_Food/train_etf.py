import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Food101Data, collate_fn
from transformers import AutoTokenizer

from tqdm import tqdm
import logging
import time
from datetime import timedelta
import random

from loss import ce_loss
from models.etf import ETF, sce_loss


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath, args):
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    logger.info(
        "\n".join(
            "%s: %s" % (k, str(v))
            for k, v in sorted(dict(vars(args)).items(), key=lambda x: x[0])
        )
    )

    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def train_warmup(args, model, optimizer_refer, dl, i_epoch):
    model.train()
    running_loss = 0.0
    with tqdm(dl, desc=f"Train Stage 1 (Warmup) Epoch: {i_epoch+1} / {args.wu_epochs}") as pbar:
        for batch in dl:
            i, t, l = batch
            i = i.to('cuda')
            t = {k: v.to('cuda') for k, v in t.items()}
            l = l.to('cuda')

            i_evi, t_evi, it_evi, i_refer_evi, t_refer_evi, it_refer_evi, *_ = model(i, t)
            
            i_tgt = (torch.argmax(i_evi, 1) == l).to(torch.long)
            loss = sce_loss(i_tgt, i_refer_evi+1, 2, args.smooth_factor).mean()
            
            t_tgt = (torch.argmax(t_evi, 1) == l).to(torch.long)
            loss += sce_loss(t_tgt, t_refer_evi+1, 2, args.smooth_factor).mean()
            
            it_tgt = (torch.argmax(it_evi, 1) == l).to(torch.long)
            loss += sce_loss(it_tgt, it_refer_evi+1, 2, args.smooth_factor).mean()
            
            running_loss += loss.clone().detach().cpu().item()
            
            optimizer_refer.zero_grad()
            loss.backward()
            optimizer_refer.step()
            
            pbar.update(1)
            pbar.set_postfix(loss=f"{running_loss/pbar.n:.4f}")
            

def train_func(args, model, optimizer_func, dl, i_epoch, stage):
    model.train()
    running_loss1, running_loss2 = 0.0, 0.0
    with tqdm(dl, desc=f"Train Stage {stage} for Functional Epoch: {i_epoch+1} / {args.epochs}") as pbar:
        for batch in dl:
            
            i, t, l = batch
            i = i.to('cuda')
            t = {k: v.to('cuda') for k, v in t.items()}
            l = l.to('cuda')

            ### 2a
            i_evi, t_evi, it_evi, *_ = model(i, t)
            loss  = ce_loss(l, i_evi+1, args.n_classes, i_epoch, args.annealing_epoch).mean()
            loss += ce_loss(l, t_evi+1, args.n_classes, i_epoch, args.annealing_epoch).mean()
            loss += ce_loss(l, it_evi+1, args.n_classes, i_epoch, args.annealing_epoch).mean()
            running_loss1 += loss.clone().detach().cpu().item()
            optimizer_func.zero_grad()
            loss.backward()
            optimizer_func.step()
        
            ### 2b
            *_, i_t_evi = model(i, t)
            loss  = ce_loss(l, i_t_evi+1, args.n_classes, i_epoch, args.annealing_epoch).mean()
            running_loss2 += loss.clone().detach().cpu().item()
            optimizer_func.zero_grad()
            loss.backward()
            optimizer_func.step()
            
            pbar.update(1)
            pbar.set_postfix(loss1=f"{running_loss1/pbar.n:.4f}", loss2=f"{running_loss2/pbar.n:.4f}")
            
        
def train_refer(args, model, optimizer_refer, dl, i_epoch, stage):
    model.train()
    running_loss = 0.0
    with tqdm(dl, desc=f"Train Stage {stage} for Refer Epoch: {i_epoch+1} / {args.epochs}") as pbar:
        for batch in dl:
            i, t, l = batch
            i = i.to('cuda')
            t = {k: v.to('cuda') for k, v in t.items()}
            l = l.to('cuda')

            *_, i_t_evi = model(i, t)
            loss = ce_loss(l, i_t_evi+1, args.n_classes, i_epoch, args.annealing_epoch).mean()
            running_loss += loss.clone().detach().cpu().item()
            optimizer_refer.zero_grad()
            loss.backward()
            optimizer_refer.step()
            
            pbar.update(1)
            pbar.set_postfix(loss=f"{running_loss/pbar.n:.4f}")


def train_stage1(args, logger, model, dl):
    
    refer_opt = optim.AdamW(
        [
            {'params': model.refer_text.parameters()},
            {'params': model.refer_img.parameters()},
            {'params': model.refer_it.parameters()},
        ],
        lr=args.rlr
    )
    logger.info("Train Stage 1 starts")
    for i_epoch in range(args.wu_epochs):
        train_warmup(args, model, refer_opt, dl, i_epoch)
    torch.save(model.state_dict(), args.ckpt_path)
    logger.info("Train_Stage 1 ends")
    
    
def train_stage2(args, logger, model, train_dl, val_dl, best_acc=0):
    
    func_opt = optim.AdamW(
        [
            {'params': model.text_encoder.parameters()},
            {'params': model.img_encoder.parameters()},
            {'params': model.text_clf.parameters()},
            {'params': model.img_clf.parameters()},
            {'params': model.it_clf.parameters()},
        ],
        lr=args.lr,
    )
    
    logger.info("Train Stage 2 starts")
    best_acc, best_epoch = best_acc, -1
    patience_count = 0
    for i_epoch in range(args.epochs):
        train_func(args, model, func_opt, train_dl, i_epoch, 2)
        
        model.eval()
        acc = 0
        with tqdm(val_dl, desc=f"Validation") as vpbar:
            for batch in val_dl:
                i, t, l = batch
                i = i.to('cuda')
                t = {k: v.to('cuda') for k, v in t.items()}
                l = l.to('cuda')
                with torch.no_grad():
                    *_, i_t_evi = model(i, t)
                pred = torch.argmax(i_t_evi, dim=1)
                corr_num = torch.sum(pred == l)
                acc += corr_num.item()
                vpbar.update(1)
            
            acc /= len(val_ds)
            if acc > best_acc:
                best_epoch = i_epoch+1
                best_acc = acc
                torch.save(model.state_dict(), args.ckpt_path)
                patience_count = 0
            else:
                patience_count += 1
            logger.info(f"acc: {acc}, best_acc: {best_acc}, best_epoch: {best_epoch}")
            vpbar.set_postfix(acc=acc, best_acc=acc, best_epoch=best_epoch)
            if patience_count >= args.patience:
                break
        
    logger.info("Train_Stage 2 ends")
    
    return best_acc


def train_stage3(args, logger, model, train_dl, val_dl, best_acc=0):
    
    refer_opt = optim.AdamW(
        [
            {'params': model.refer_text.parameters()},
            {'params': model.refer_img.parameters()},
            {'params': model.refer_it.parameters()},
        ],
        lr=args.rlr,
    )
    
    logger.info("Train Stage 3 starts")
    best_acc, best_epoch = best_acc, -1
    patience_count = 0
    for i_epoch in range(args.epochs):
        train_refer(args, model, refer_opt, train_dl, i_epoch, 3)
        
        model.eval()
        acc = 0
        with tqdm(val_dl, desc=f"Validation") as vpbar:
            for batch in val_dl:
                i, t, l = batch
                i = i.to('cuda')
                t = {k: v.to('cuda') for k, v in t.items()}
                l = l.to('cuda')
                with torch.no_grad():
                    *_, i_t_evi = model(i, t)
                pred = torch.argmax(i_t_evi, dim=1)
                corr_num = torch.sum(pred == l)
                acc += corr_num.item()
                vpbar.update(1)
            
            acc /= len(val_ds)
            if acc > best_acc:
                best_epoch = i_epoch+1
                best_acc = acc
                torch.save(model.state_dict(), args.ckpt_path)
                patience_count = 0
            else:
                patience_count += 1
            logger.info(f"acc: {acc}, best_acc: {best_acc}, best_epoch: {best_epoch}")
            vpbar.set_postfix(acc=acc, best_acc=best_acc, best_epoch=best_epoch)
            if patience_count >= args.patience:
                break
        
    logger.info("Train_Stage 3 ends")
    
    return best_acc


def train_stage4(args, logger, model, train_dl, val_dl, best_acc=0):
    
    func_opt = optim.AdamW(
        [
            {'params': model.text_encoder.parameters()},
            {'params': model.img_encoder.parameters()},
            {'params': model.text_clf.parameters()},
            {'params': model.img_clf.parameters()},
            {'params': model.it_clf.parameters()},
        ],
        lr=args.lr,
    )
    
    logger.info("Train Stage 4 starts")
    best_acc, best_epoch = best_acc, -1
    patience_count = 0
    for i_epoch in range(args.epochs):
        train_func(args, model, func_opt, train_dl, i_epoch, 4)
        
        model.eval()
        acc = 0
        with tqdm(val_dl, desc=f"Validation") as vpbar:
            for batch in val_dl:
                i, t, l = batch
                i = i.to('cuda')
                t = {k: v.to('cuda') for k, v in t.items()}
                l = l.to('cuda')
                with torch.no_grad():
                    *_, i_t_evi = model(i, t)
                pred = torch.argmax(i_t_evi, dim=1)
                corr_num = torch.sum(pred == l)
                acc += corr_num.item()
                vpbar.update(1)
            
            acc /= len(val_ds)
            if acc > best_acc:
                best_epoch = i_epoch+1
                best_acc = acc
                torch.save(model.state_dict(), args.ckpt_path)
                patience_count = 0
            else:
                patience_count += 1
            logger.info(f"acc: {acc}, best_acc: {best_acc}, best_epoch: {best_epoch}")
            vpbar.set_postfix(acc=acc, best_acc=best_acc, best_epoch=best_epoch)
            if patience_count >= args.patience:
                break
        
    logger.info("Train_Stage 4 ends")
    
    return best_acc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--v-num", type=int, default=0)
    parser.add_argument("--rlr", type=float, default=5e-6)
    args = parser.parse_args()
    
    # set_seed(args.v_num)
    
    args.model_name = "ETF"
    
    args.split_file = "dataset/Food101/split.json"
    args.text_file = "dataset/Food101/text.json"
    args.class_file = "dataset/Food101/class_idx.json"
    args.data_path = "dataset/Food101/"
    
    args.lm_name = "bert-base-uncased"
    args.max_text_len = 128
    args.n_classes = 101
    
    args.debug=False
    args.batch_size = 64
    args.lr = 1e-5
    args.epochs = 50
    # args.rlr = 1e-6
    args.wu_epochs = 1
    args.smooth_factor = 0.9
    args.annealing_epoch = 10
    args.patience = 5
    
    args.ckpt_path = f'ckpts/{args.model_name}_Food101_v{args.v_num}_best_ckpt.pt'
    logger = create_logger(f'logs/{args.model_name}_Food101_v{args.v_num}', args)

    # dataset and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    
    args.split = "train"
    tr_ds = Food101Data(args)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size,
                       shuffle=True, num_workers=4, collate_fn=lambda x: collate_fn(args, x, tokenizer))
    args.split = "val"
    val_ds = Food101Data(args)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, collate_fn=lambda x: collate_fn(args, x, tokenizer))
    args.split = "test"
    te_ds = Food101Data(args)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size,
                       shuffle=False, num_workers=4, collate_fn=lambda x: collate_fn(args, x, tokenizer))
    ###########################
    
    model = ETF(args)
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Stage 1
    train_stage1(args, logger, model, tr_dl)
    best_ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(best_ckpt)
    model.to('cuda')
    
    # Stage 2
    best_val_acc = train_stage2(args, logger, model, tr_dl, val_dl, best_acc=0)
    best_ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(best_ckpt)
    model.to('cuda')
    
    # Stage 3
    best_val_acc = train_stage3(args, logger, model, tr_dl, val_dl, best_acc=best_val_acc)
    best_ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(best_ckpt)
    model.to('cuda')
    
    # Stage 4
    best_val_acc = train_stage4(args, logger, model, tr_dl, val_dl, best_acc=best_val_acc)
    best_ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(best_ckpt)
    model.to('cuda')
    
    logger.info(best_val_acc)
    acc = 0
    for batch in te_dl:
        i, t, l = batch
        i = i.to('cuda')
        t = {k: v.to('cuda') for k, v in t.items()}
        l = l.to('cuda')
        with torch.no_grad():
            *_, i_t_evi = model(i, t)
        pred = torch.argmax(i_t_evi, dim=1)
        corr_num = torch.sum(pred == l)
        acc += corr_num.item()
    logger.info(f"test acc: {acc / len(te_ds)}")
