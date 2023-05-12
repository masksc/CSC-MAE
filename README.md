# Contrastive Single-cell Masked Autoencoder for Clustering of scRNA-seq Data

Official implementation for "Contrastive Single-cell Masked Autoencoder for Clustering of scRNA-seq Data".

CSC-MAE is a new contrastive self-supervised framework that can capture interconnections across gene expression matrix (multiple groups of expression sequences) by masked expression modeling.

<div align="center">
  <img width="100%" alt="CSC-MAE Illustration" src="assets/framework.png">
</div>


## Environment

To setup the environment, please simply run

```
pip install -r requirements.txt
```

## Datasets

###  scRNA-seq 

Data can be downloaded from [here](https://drive.google.com/file/d/1JKcLwZypAk8JIn44jt8DVj3wpU_ag2Wd/view?usp=sharing)

**Notice:** please update the path of data (--path) in main.py accordingly.



## Train & Test

For training and testing on 10X_PBMC dataset, please run

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
    --model cscmae_vit_base_patch4 \
    --batch_size 128 \
    --accum_iter 1 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /path/to/single-cell_data/ \
    --feat_path /path/to/sequence_data_feature/ \
    --dataset 10X_PBMC \
    --nb_genes 1296 \
    --input_size 64 \
    --output_dir /path/to/output_dir  \
    --log_dir /path/to/log_dir  \
    --use_feat_target \
    --dist_eval
```

