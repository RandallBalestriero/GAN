import sys
sys.path.insert(0, "../../TheanoXLA")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients, jacobians
from theanoxla.utils import batchify, vq_to_boundary
import theanoxla
from sklearn import datasets
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.special import gamma


lrs = theanoxla.schedules.PiecewiseConstant(1, {10: 2, 20:3})

for run in range(30):

    # reset variables
    lrs.reset()

    # training
    for epoch in range(30):
        lrs.update()
        print(lrs.value.get({}))
