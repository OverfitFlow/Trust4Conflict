import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.tmc import TMC
from data import Food101Data, collate_fn
from transformers import AutoTokenizer
from loss import ce_loss
from tqdm import tqdm
import logging
import time
from datetime import timedelta
import random


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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--v-num", type=int, default=0)
    args = parser.parse_args()
    
    # set_seed(args.v_num)
    
    args.model_name = "TMC"
    
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
    
    ckpt_path = f'ckpts/{args.model_name}_Food101_v{args.v_num}_best_ckpt.pt'
    args.ckpt_path = ckpt_path
    logger = create_logger(f'logs/{args.model_name}_Food101_v{args.v_num}', args)

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
    # print(tr_ds.split, len(tr_ds))
    # print(te_ds.split, len(te_ds))
    
    patience = 5
    patience_count = 0
    
    model = TMC(args)
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    
    best_epoch, best_acc = -1, -1
    for epoch in range(args.epochs):

            model.train()
            running_loss = 0.0
            with tqdm(tr_dl, desc=f"Training Epoch: {epoch+1} / {args.epochs}") as pbar:
                for batch in pbar:

                    i, t, l = batch
                    i = i.to('cuda')
                    t = {k: v.to('cuda') for k, v in t.items()}
                    l = l.to('cuda')

                    i_evi, t_evi, i_t_evi = model(i, t)
                    
                    i_t_alpha = i_t_evi + 1
                    loss = ce_loss(l, i_t_alpha, args.n_classes, epoch+1, 10)
                    i_alpha = i_evi + 1
                    loss += ce_loss(l, i_alpha, args.n_classes, epoch+1, 10)
                    t_alpha = t_evi + 1
                    loss += ce_loss(l, t_alpha, args.n_classes, epoch+1, 10)
                    loss = loss.mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.cpu().item()
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{running_loss/pbar.n:.4f}")
                
            model.eval()
            acc = 0
            with tqdm(val_dl, desc=f"Validation Epoch: {epoch+1} / {args.epochs}") as pbar2:
                for batch in val_dl:
                    i, t, l = batch
                    i = i.to('cuda')
                    t = {k: v.to('cuda') for k, v in t.items()}
                    l = l.to('cuda')
                    with torch.no_grad():
                        i_evi, t_evi, i_t_evi = model(i, t)
                    pred = torch.argmax(i_t_evi, dim=1)
                    corr_num = torch.sum(pred == l)
                    acc += corr_num.item()
                    pbar2.update(1)
                    
                acc /= len(val_ds)
                if acc > best_acc:
                    best_epoch = epoch+1
                    best_acc = acc
                    torch.save(model.state_dict(), ckpt_path)
                    patience_count = 0
                else:
                    patience_count += 1
                    
                pbar2.set_postfix(acc=acc, best_acc=best_acc, best_epoch=best_epoch)

                if patience_count == patience:
                    break

    model.eval()
    best_ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(best_ckpt)
    model.to('cuda')
    
    acc = 0
    for batch in te_dl:
        i, t, l = batch
        i = i.to('cuda')
        t = {k: v.to('cuda') for k, v in t.items()}
        l = l.to('cuda')
        with torch.no_grad():
            i_evi, t_evi, i_t_evi = model(i, t)
        pred = torch.argmax(i_t_evi, dim=1)
        corr_num = torch.sum(pred == l)
        acc += corr_num.item()
    logger.info(f"test acc: {acc / len(te_ds)}")
