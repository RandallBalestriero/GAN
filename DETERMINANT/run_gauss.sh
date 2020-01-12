#!/bin/bash

for dataset in 0 1 2
do
    python multigauss_determinant.py --dataset $dataset --WG 32
done



