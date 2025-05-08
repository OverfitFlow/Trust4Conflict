python train.py \
        --log-file etf-caltech-normal \
        --data-name Caltech101 \
        --model-name ETF \
        --lr 1e-4 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs 1 \
        --rlr 3e-5