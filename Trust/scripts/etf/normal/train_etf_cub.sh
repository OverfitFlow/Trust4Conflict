python train.py \
        --log-file etf-cub-normal \
        --data-name CUB \
        --model-name ETF \
        --lr 1e-3 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs 1 \
        --rlr 3e-4
