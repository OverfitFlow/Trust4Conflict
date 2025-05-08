python train.py \
        --log-file etf-pie-normal \
        --data-name PIE \
        --model-name ETF \
        --lr 3e-3 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs 1 \
        --rlr 1e-3
