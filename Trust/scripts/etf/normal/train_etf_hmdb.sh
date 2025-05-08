python train.py \
        --log-file etf-hmdb-normal \
        --data-name HMDB \
        --model-name ETF \
        --lr 3e-4 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs 1 \
        --rlr 1e-4
