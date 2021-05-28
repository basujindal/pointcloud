import h5py
import os
import random
from torch.utils.data import Dataset
import numpy as np
import pickle
import torch.nn as nn
import math
import torch
from pytorch3d.loss import chamfer_distance
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch3d
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

class PCDataLoader(Dataset):
    def __init__(self, points, gt,labels, n_points=2048):
        self.n_points = n_points
        self.points = points
        self.gt = gt
        self.labels = labels

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):

        pc = self.points[index][:,0:3]
        gt = self.gt[index][:,0:3]
        labels = self.labels[index]
        return pc, gt, labels

