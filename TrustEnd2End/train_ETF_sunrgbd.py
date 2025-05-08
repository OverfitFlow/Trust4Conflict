import argparse
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms
from data.aligned_conc_dataset import AlignedConcDataset
from utils.utils import *
from utils.logger import create_logger
import os
from torch.utils.data import DataLoader

from models.ETF import ETF
from models.util import sce_loss, ce_loss

"""

B train_func two stages different parameters set
stage 2b no encoder parameters
"""


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="./datasets/sunrgbd/")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="ReleasedVersion")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--savedir", type=str, default="./savepath/ETF/sunrgbd")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=19)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--belief_fusion", type=str, default="bcf", choices=["bcf", "a-cbf", "abf"])
    parser.add_argument("--smooth_factor", type=float, default=0.9)
    parser.add_argument("--rlr", type=float, default=2e-4)
    parser.add_argument("--rlr_factor", type=float, default=0.3)
    parser.add_argument("--rlr_patience", type=int, default=10)
    parser.add_argument("--warmup_max_epochs", type=int, default=1)
    

def model_eval(model, dl):
    model.eval()
    with torch.no_grad():
        depth_preds, rgb_preds, depthrgbps_preds, tgts = [], [], [], []
        for batch in dl:
            
            rgb, depth, tgt = batch['A'], batch['B'], batch['label']
            rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
            
            *_, depth_alpha, rgb_alpha, _, depth_rgb_ps_alpha = model(rgb, depth)

            depth_pred = depth_alpha.argmax(dim=1).cpu().detach().numpy()
            rgb_pred = rgb_alpha.argmax(dim=1).cpu().detach().numpy()
            depth_rgb_ps_pred = depth_rgb_ps_alpha.argmax(dim=1).cpu().detach().numpy()

            depth_preds.extend(depth_pred.tolist())
            rgb_preds.extend(rgb_pred.tolist())
            depthrgbps_preds.extend(depth_rgb_ps_pred.tolist())
            tgt = tgt.cpu().detach().numpy()
            tgts.extend(tgt.tolist())

    metrics = {}
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgbps_preds)
    return metrics


def train_warmup(args, model, dl, optimizer_refer, global_step):
    
    model.train()
    optimizer_refer.zero_grad()
    
    for batch in tqdm(dl, total=len(dl)):
        
        rgb, depth, tgt = batch['A'], batch['B'], batch['label']
        rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
        
        depth_func_alpha, rgb_func_alpha, ps_func_alpha, depth_refer_alpha, rgb_refer_alpha, ps_refer_alpha, *_ = model(rgb, depth)
        
        rgb_t = (torch.argmax(rgb_func_alpha, 1) == tgt).to(torch.long)
        loss = sce_loss(rgb_t, rgb_refer_alpha, 2, args.smooth_factor)
        depth_t = (torch.argmax(depth_func_alpha, 1) == tgt).to(torch.long)
        loss += sce_loss(depth_t, depth_refer_alpha, 2, args.smooth_factor)
        ps_t = (torch.argmax(ps_func_alpha, 1) == tgt).to(torch.long)
        loss += sce_loss(ps_t, ps_refer_alpha, 2, args.smooth_factor)
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer_refer.step()
            optimizer_refer.zero_grad()

    return global_step


def train_func(args, model, optimizer_func, dl, i_epoch, global_step):

    model.train()
    optimizer_func.zero_grad()
    
    batch_caches = []
    for batch in tqdm(dl, total=len(dl)):
        
        batch_caches.append(batch)
        
        rgb, depth, tgt = batch['A'], batch['B'], batch['label']
        rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
        
        depth_func_alpha, rgb_func_alpha, ps_func_alpha, *_ = model(rgb, depth)
        
        loss = ce_loss(tgt, rgb_func_alpha, args.n_classes, i_epoch, args.annealing_epoch)
        loss += ce_loss(tgt, depth_func_alpha, args.n_classes, i_epoch, args.annealing_epoch)
        loss += ce_loss(tgt, ps_func_alpha, args.n_classes, i_epoch, args.annealing_epoch)
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer_func.step()
            optimizer_func.zero_grad()
    
            for batch in batch_caches:
                rgb, depth, tgt = batch['A'], batch['B'], batch['label']
                rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
                
                *_, depth_rgb_alpha = model(rgb, depth)
                loss = ce_loss(tgt, depth_rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                
            optimizer_func.step()
            optimizer_func.zero_grad()
            batch_caches.clear()
        
    return global_step
    

def train_refer(args, model, optimizer_refer, dl, i_epoch, global_step):
    
    model.train()
    optimizer_refer.zero_grad()

    for batch in tqdm(dl, total=len(dl)):
                
        rgb, depth, tgt = batch['A'], batch['B'], batch['label']
        rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
        
        *_, depth_rgb_alpha = model(rgb, depth)
        loss = ce_loss(tgt, depth_rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch)
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer_refer.step()
            optimizer_refer.zero_grad()
            
    return global_step


def train_stage1(args, logger, model, train_dl):
    # stage 1
    refer_optim = optim.Adam(
        [
            {'params': model.refer_depth.parameters()},
            {'params': model.refer_rgb.parameters()},
            {'params': model.refer_ps.parameters()},
        ],
        lr=args.rlr,
        weight_decay=1e-5
    )
    
    logger.info("Train Stage 1 starts")
    global_step = 0
    
    for i_epoch in range(args.warmup_max_epochs):
        global_step = train_warmup(args, model, train_dl, refer_optim, global_step)
    save_checkpoint(
        {
            "epoch": i_epoch + 1,
            "stage": 1,
            "state_dict": model.state_dict(),
        },
        True,
        args.savedir,
    )
    logger.info("Train Stage 1 ends")
    

def train_stage2(args, logger, model, train_dl, test_dl, last_best_metric=-np.inf):
    # stage 2
    func_optim = optim.Adam(
        [
            {'params': model.depthenc.parameters()},
            {'params': model.rgbenc.parameters()},
            {'params': model.clf_depth.parameters()},
            {'params': model.clf_rgb.parameters()},
            {'params': model.clf_ps.parameters()},
        ],
        lr=args.lr,
        weight_decay=1e-5
    )
    func_sche = optim.lr_scheduler.ReduceLROnPlateau(
        func_optim, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    
    logger.info("Train Stage 2 starts")
    global_step, best_metric, n_no_improve = 0, last_best_metric, 0
    
    for i_epoch in range(args.max_epochs):
        global_step = train_func(args, model, func_optim, train_dl, i_epoch+1, global_step)
        
        metrics = model_eval(model, test_dl)
        log_metrics(f"Stage 2 Test {i_epoch+1}/{args.max_epochs}", metrics, logger)
        
        tuning_metric = metrics["depthrgb_acc"]
        func_sche.step(tuning_metric)

        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1
            
        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "stage": 2,
                "state_dict": model.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break
        
    logger.info("Train Stage 2 ends")


def train_stage3(args, logger, model, train_dl, test_dl, last_best_metric=-np.inf):
    # stage 3
    refer_optim = optim.Adam(
        [
            {'params': model.refer_depth.parameters()},
            {'params': model.refer_rgb.parameters()},
            {'params': model.refer_ps.parameters()},
        ],
        lr=args.rlr,
        weight_decay=1e-5
    )
    refer_sche = optim.lr_scheduler.ReduceLROnPlateau(
        refer_optim, "max", patience=args.rlr_patience, verbose=True, factor=args.rlr_factor
    )
    
    global_step, best_metric, n_no_improve = 0, last_best_metric, 0
    logger.info("Train Stage 3 starts")
    
    for i_epoch in range(args.max_epochs):
        global_step = train_refer(args, model, refer_optim, train_dl, i_epoch+1, global_step)
        
        metrics = model_eval(model, test_dl)
        log_metrics(f"Stage 3 Test {i_epoch+1}/{args.max_epochs}", metrics, logger)
        
        tuning_metric = metrics["depthrgb_acc"]
        refer_sche.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1
            
        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "stage": 3,
                "state_dict": model.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break
    logger.info("Train Stage 3 ends")


def train_stage4(args, logger, model, train_dl, test_dl, last_best_metric=-np.inf):
    # stage 4
    func_optim = optim.Adam(
        [
            {'params': model.depthenc.parameters()},
            {'params': model.rgbenc.parameters()},
            {'params': model.clf_depth.parameters()},
            {'params': model.clf_rgb.parameters()},
            {'params': model.clf_ps.parameters()},
        ],
        lr=args.lr,
        weight_decay=1e-5
    )
    func_sche = optim.lr_scheduler.ReduceLROnPlateau(
        func_optim, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    
    global_step, best_metric, n_no_improve = 0, last_best_metric, 0
    logger.info("Train Stage 4 starts")
    
    for i_epoch in range(args.max_epochs):
        global_step = train_func(args, model, func_optim, train_dl, i_epoch+1, global_step)
        
        metrics = model_eval(model, test_dl)
        log_metrics(f"Stage 4 Test {i_epoch+1}/{args.max_epochs}", metrics, logger)
        
        tuning_metric = metrics["depthrgb_acc"]
        func_sche.step(tuning_metric)

        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1
            
        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "stage": 4,
                "state_dict": model.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break
    logger.info("Train Stage 4 ends")


def train(args):
    
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    
    if args.seed is not None:
        logger.warn(f"Setting seed with number {args.seed} for non-determinstic results")
        set_seed(args.seed)
    else:
        logger.warn(f"Not using seed")
    
    train_transforms = list()
    train_transforms.append(transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)))
    train_transforms.append(transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=[0.6983, 0.3918, 0.4474], std=[0.1648, 0.1359, 0.1644]))
    val_transforms = list()
    val_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=[0.6983, 0.3918, 0.4474], std=[0.1648, 0.1359, 0.1644]))

    train_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'), transform=transforms.Compose(train_transforms)),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers)
    test_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'), transform=transforms.Compose(val_transforms)),
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers)
    
    model = ETF(args)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    
    if args.warmup_max_epochs != 0:
        train_stage1(args, logger, model, train_loader)
        load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    else:
        logger.info("skipping warmup stage as warmup max epochs equal to 0")
    
    torch.cuda.empty_cache()
    train_stage2(args, logger, model, train_loader, test_loader)
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    test_metrics = model_eval(model, test_loader)
    log_metrics(f"Stage 2 best", test_metrics, logger)
    
    torch.cuda.empty_cache()
    train_stage3(args, logger, model, train_loader, test_loader, test_metrics["depthrgb_acc"])
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    test_metrics = model_eval(model, test_loader)
    log_metrics(f"Stage 3 best", test_metrics, logger)
    
    torch.cuda.empty_cache()
    train_stage4(args, logger, model, train_loader, test_loader, test_metrics["depthrgb_acc"])
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    test_metrics = model_eval(model, test_loader)
    log_metrics(f"Stage 4 best", test_metrics, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
