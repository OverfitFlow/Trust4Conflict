values="0 2 5 10"

for value in $values
do
python train.py \
        --log-file etf-scene-normal-wu$value \
        --data-name Scene15 \
        --model-name ETF \
        --lr 1e-2 \
        --epochs 500 \
        --annealing-step 10 \
        --smooth-factor 0.9 \
        --warm-epochs $value \
        --rlr 3e-3
done