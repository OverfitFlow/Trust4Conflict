To run our codes, for example,
get the results as reported in our paper for ETF,

```shell
sh scripts/etf/normal/train_etf_scene.sh
```
All hyperparameters used for different datasets are given in their corresponding files,
for example, for HMDB dataset, run
```shell
sh scripts/etf/normal/train_etf_hmdb.sh
```

To run the TF, another variation,
```shell
sh scripts/tf/normal/train_tf_scene.sh
```
Whatever the method is, either TF or ETF, for same dataset, the hyper-parameters we used are same,
so we skip scripts for TF on other datasets.

To simulate the conflict settings by injecting Guassian noises, run
```shell
sh scripts/etf/simulate_conflict/train_etf_scene.sh
```

To compare different warm-up epochs, run
```shell
sh scripts/etf/warmup/train_etf_scene.sh
```