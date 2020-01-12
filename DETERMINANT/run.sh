#!/bin/bash
GPU=0
for D in 2
do
    GPU=$(((GPU+1)%3))
    for std in 0.002 0.01 0.05 0.1 0.3 1 2
    do 
        screen -dmS det$D$std$radius$run bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python comparison.py --D $D --WG 64 --radius 1 --std $std";
    done
done



