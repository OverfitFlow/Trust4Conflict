import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Multi_view_data, load_data
from models import ETF, TF
from util import get_args, get_logger, test, train_func, train_warmup, train_refer, test_final


torch.cuda.empty_cache()


if __name__ == "__main__":
    
    args = get_args()
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = get_logger(args)
    
    ckpt_path = f"ckpts/{args.model_name}/{args.data_name}/warmup_{args.warm_epochs}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if args.conflict_test:
        ckpt_path = f"{ckpt_path}/Conflict_{args.conflict_sigma}"
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
    else:
        ckpt_path = f"{ckpt_path}/Normal"
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
    
    args.used_views = [int(each) for each in args.used_views]
    # print(args.used_views, type(args.used_views))
    
    all_accs = []
    all_c1s = []
    all_uncers = []
    all_kappas = []
    
    n_repeat = 10
    for s in range(n_repeat):
        args.conflict_seed = s
        
        # split data
        data, split_idx = load_data(args)
        train_data = Multi_view_data(args, data, split_idx["train_idx"], split_idx["test_idx"], is_train=True, used_views=args.used_views)
        train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_data = Multi_view_data(args, data, split_idx["train_idx"], split_idx["test_idx"],  is_train=False, used_views=args.used_views)
        test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        
        args.n_views = train_data.n_views
        args.views_dims = train_data.views_dims
        args.n_classes = train_data.n_classes
        logger.info(args)
        
        if args.model_name == "ETF":
            logger.info("Using ETF")
            model_cls = ETF
        elif args.model_name == "TF":
            logger.info("Using TF")
            model_cls = TF
        else:
            raise NotImplementedError
        m = model_cls(args)
        save_path = f'./{ckpt_path}/ckpt_{s}.pt'
        m.init_refer_net()  # always first inference style
        m.cuda()
        
        ####1
        refer_opt = optim.Adam(m.refer_net.parameters(), lr=args.rlr, weight_decay=1e-4)
        for epoch in range(1, 1+args.warm_epochs):
            train_warmup(args, train_dl, m, None, refer_opt, None, epoch)
        logger.info("TF Stage1 (Warmup) done.")

        #####2
        func_opt = optim.Adam(m.func_net.parameters(), lr=args.lr, weight_decay=1e-4)
        tf_best_epoch, tf_best_acc, tf_best_loss = -1, -1, -1
        for epoch in range(1, 1+args.epochs):
            train_func(args, train_dl, m, func_opt, None, None, epoch)
            test_loss, test_acc = test(args, test_dl, m, epoch)
            # logger.info("TF Test1 Epoch {} Loss {:.4f} Acc {:.4f}".format(epoch, test_loss, test_acc))
            if tf_best_acc < test_acc:
                tf_best_epoch = epoch
                tf_best_acc = test_acc
                tf_best_loss = test_loss
                torch.save(
                    {
                        'model_state_dict': m.state_dict(),
                        'best_epoch': tf_best_epoch
                    },
                    save_path)
        logger.info("TF Stage2 Best Epoch {} Loss {:.4f} Acc {:.4f}".format(tf_best_epoch, tf_best_loss, tf_best_acc))
        m.cpu()
        ckpt = torch.load(save_path, map_location='cpu')
        m.load_state_dict(ckpt["model_state_dict"])
        m.cuda()
        test_loss, test_acc = test(args, test_dl, m, epoch)
        logger.info("TF Stage2 Loadback Epoch {} Loss {:.4f} Acc {:.4f}".format(ckpt['best_epoch'], test_loss, test_acc))
        
        ####3
        refer_opt = optim.Adam(m.refer_net.parameters(), lr=args.rlr, weight_decay=1e-4)
        tf_best_epoch, tf_best_acc, tf_best_loss = -1, -1, -1
        for epoch in range(1, 1+args.epochs):
            train_refer(args, train_dl, m, None, refer_opt, None, epoch)
            test_loss, test_acc = test(args, test_dl, m, epoch)
            # logger.info("TF Test2 Epoch {} Loss {:.4f} Acc {:.4f}".format(epoch, test_loss, test_acc))
            if tf_best_acc < test_acc:
                tf_best_epoch = epoch
                tf_best_acc = test_acc
                tf_best_loss = test_loss
                torch.save(
                    {
                        'model_state_dict': m.state_dict(),
                        'best_epoch': tf_best_epoch
                    },
                    save_path)
        logger.info("TF Stage3 Best Epoch {} Loss {:.4f} Acc {:.4f}".format(tf_best_epoch, tf_best_loss, tf_best_acc))
        m.cpu()
        ckpt = torch.load(save_path, map_location='cpu')
        m.load_state_dict(ckpt["model_state_dict"])
        m.cuda()
        test_loss, test_acc = test(args, test_dl, m, epoch)
        logger.info("TF Stage3 Loadback Epoch {} Loss {:.4f} Acc {:.4f}".format(ckpt['best_epoch'], test_loss, test_acc))
        
        #####4
        func_opt = optim.Adam(m.func_net.parameters(), lr=args.lr, weight_decay=1e-4)
        tf_best_epoch, tf_best_acc, tf_best_loss = -1, -1, -1
        for epoch in range(1, 1+args.epochs):
            train_func(args, train_dl, m, func_opt, None, None, epoch)
            test_loss, test_acc = test(args, test_dl, m, epoch)
            # logger.info("TF Test1 Epoch {} Loss {:.4f} Acc {:.4f}".format(epoch, test_loss, test_acc))
            if tf_best_acc < test_acc:
                tf_best_epoch = epoch
                tf_best_acc = test_acc
                tf_best_loss = test_loss
                torch.save(
                    {
                        'model_state_dict': m.state_dict(),
                        'best_epoch': tf_best_epoch
                    },
                    save_path)
        logger.info("TF Stage4 Best Epoch {} Loss {:.4f} Acc {:.4f}".format(tf_best_epoch, tf_best_loss, tf_best_acc))
        m.cpu()
        
        ckpt = torch.load(save_path, map_location='cpu')
        m.load_state_dict(ckpt["model_state_dict"])
        m.cuda()
        test_loss, test_acc, test_uauroc, test_kappa, test_c1 = test_final(args, test_dl, m, ckpt['best_epoch'])
        logger.info("TF Stage4 Loadback Epoch {} Loss {:.4f} Acc {:.4f} UncerAuRoC {:.4f} Kappa {:.4f} C1 {:.4f}".format(
            ckpt['best_epoch'], test_loss, test_acc, test_uauroc, test_kappa, test_c1))
        
        all_accs.append(test_acc)
        all_uncers.append(test_uauroc)
        all_kappas.append(test_kappa)
        all_c1s.append(test_c1)
        
    logger.info(f"{len(all_accs)} {np.mean(all_accs):.4f}±{np.std(all_accs):.4f} \
{np.mean(all_uncers):.4f}±{np.std(all_uncers):.4f} \
{np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f} \
{np.mean(all_c1s):.4f}±{np.std(all_c1s):.4f}")
