import sys
# sys.path.append("..")
from sklearn import metrics
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from collections import Counter
from sklearn.manifold import TSNE
import train
import random

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", default='../../../data/single-cell/all_data/', help="path to data")
parser.add_argument("--nb_genes", default=1296, type=int, help="number of keep genes")
parser.add_argument("--dataset", default='Quake_Smart-seq2_Trachea', help="which dataset to use")

args = parser.parse_args()

run =0
torch.manual_seed(run)
torch.cuda.manual_seed_all(run)
np.random.seed(run)
random.seed(run)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# plt.ion()
# plt.show()
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# path= "../../../data/single-cell/all_data/"
# # check available files
# !ls ../real_data


# select dataset to analyze

data_mat = h5py.File(f"{args.path}real_data/{args.dataset}.h5", "r")
X = np.array(data_mat['X'])
Y = np.array(data_mat['Y'])
print('X:', X.shape)         # (4271, 16653)
print('Y:', Y.shape)         # (4271, )
print(np.where(X ==0)[0].shape[0]/(X.shape[0]*X.shape[1]))
cluster_number = np.unique(Y).shape[0]
nb_genes = args.nb_genes  # 1296 # 4096 # 6400 # 10000


X = train.preprocess(X, nb_genes = nb_genes)
print('After Preprocess X:', X.shape)

results = train.run(X,
                     cluster_number,
                     args.dataset,
                     Y=Y,
                     nb_epochs=30,
                     layers=[200, 40, 60],
                     dropout = 0.9,
                     save_pred = True,
                     cluster_methods =["KMeans"])

df = pd.DataFrame(columns = ["dataset", "pred", "features"])

print('embedding:', results['features'].shape)         # (4271, 60)
print('pred:', results['kmeans_pred'].shape)           # (4271, )
df.loc[df.shape[0]] = [args.dataset, results['kmeans_pred'], results['features']]

df.to_pickle(f"./output/pickle_results/real_data/real_data_contrastivesc.pkl")

np.save(f'./real_data_feat/{args.dataset}_{nb_genes}.npy', results['features'])
np.save(f'./real_data_pred/{args.dataset}_{nb_genes}.npy', results['kmeans_pred'])


print(f"Kmeans clustering ARI:{round(results['kmeans_ari'], 5)}\n"+
          f"NMI:{round(float(results['kmeans_nmi']),5)}, Sil:{round(float(results['kmeans_sil']),5)}, "+
          f"Calinski:{round(float(results['kmeans_cal']),5)}")