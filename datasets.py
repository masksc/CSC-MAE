import os
import numpy as np
import scanpy.api as sc

import h5py
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset

import utils


def preprocess(X, nb_genes = 500):
    """
    Preprocessing phase as proposed in scanpy package.
    Keeps only nb_genes most variable genes and normalizes
    the data to 0 mean and 1 std.
    Args:
        X ([type]): [description]
        nb_genes (int, optional): [description]. Defaults to 500.
    Returns:
        [type]: [description]
    """
    X = np.ceil(X).astype(np.int)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    orig_X = X.copy()
    adata = sc.AnnData(X)

    adata = utils.normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X


class DatasetFolder(Dataset):

    """
    Args:
        root (string): Root directory path.
        num_region_per_sample (integer): number of regions for each sample
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.Normalize`` for regions.
     Attributes:
        samples (list): List of (sample, cell_index) tuples
        targets (list): The cell_index value for each sample in the dataset
    """

    def __init__(
            self,
            root: str,
            feat_root: str,
            dataset: str,
            nb_genes: int,
    ) -> None:
        super(DatasetFolder, self).__init__()

        self.root = root
        self.dataset = dataset

        # load raw data
        data_mat = h5py.File(f"{root}real_data/{dataset}.h5", "r")
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        print('X:', X.shape)         # (4271, 16653)
        print('Y:', Y.shape)         # (4271, )
        print(np.where(X ==0)[0].shape[0]/(X.shape[0]*X.shape[1]))
        self.cluster_number = np.unique(Y).shape[0]
        self.nb_genes = nb_genes
        
        X = preprocess(X, nb_genes = nb_genes)

        print("len_samples:", len(X))
        print("X:", X.shape)

        self.samples = X
        self.targets = Y

        # load sequence-trained features
        if nb_genes in [4097, 6401, 10001]:
            self.features = np.load(f"{feat_root}/{dataset}_{nb_genes-1}.npy")
        else:
            self.features = np.load(f"{feat_root}/{dataset}_{nb_genes}.npy")

        if self.samples.shape[1] == 6401:
            self.samples = self.samples[:,:6400]
        elif self.samples.shape[1] == 4097:
            self.samples = self.samples[:,:4096]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is cell_index of the target cell.
        """
        sample = self.samples[index].reshape(-1, int(self.nb_genes**(1/2)), int(self.nb_genes**(1/2)))
        target = self.targets[index]
        feature = self.features[index]

        return sample, target, feature

    def __len__(self) -> int:
        return len(self.samples)
