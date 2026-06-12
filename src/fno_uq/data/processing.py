import torch
import torch.nn as nn

class GaussianNormaliser():
    """
    Implementation notes:
    This should be fit to the training dataset (incrementally)
    Will be applied to the inputs a and the outputs u
    """

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None,):
        """
        example of dim: dim=[0, 2, 3] for input data shape (B, d_u, nx, ny).
        then the normalisation (standardisation happens independently for each
        channel as they may have vastly different scalings)
        """
        pass

    def partial_fit(self, data_batch):
        """
        Incrementally update statistics with a new batch.
        """
        pass

    def _update_mean_std(self, data_batch):
        """
        """
        pass

    def _incremental_update_mean_std(self, data_batch):
        """
        Welford-style incremental update using the identity
        """
        pass


    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        return (data * (self.std + self.eps)) + self.mean
    

class DataProcessor():
    """
    Decide on how best to build this into current architecture
    """

    def __init__(self, in_normaliser=None, out_normaliser=None):
        pass

