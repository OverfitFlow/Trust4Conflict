values="100 1000 10000"

for value in $values
do
    python train.py \
        --log-file etf-scene-conflict \
        --data-name Scene15 \
        --model-name ETF \
        --lr 1e-2 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs 1 \
        --rlr 3e-3 \
        --conflict-test \
        --conflict-sigma $value
done