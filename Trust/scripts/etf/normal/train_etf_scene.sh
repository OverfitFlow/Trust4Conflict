python train.py \
        --log-file etf-scene-normal \
        --data-name Scene15 \
        --model-name ETF \
        --lr 1e-2 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs 1 \
        --rlr 3e-3
