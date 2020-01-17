#!/bin/bash
for D in 2
do
    GPU=0
    for std in 0.002 0.01 0.05 0.1 0.3 1 2
    do 
        screen -dmS det$D$std$radius$run bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python comparison.py --D $D --WG 64 --radius 0 --std $std --scale 1";
    done
done

for D in 2
do
    GPU=1
    for std in 0.002 0.01 0.05 0.1 0.3 1 2
    do 
        screen -dmS det$D$std$radius$run bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python comparison.py --D $D --WG 64 --radius 0 --std 0.3 --scale $std";
    done
done

for D in 2
do
    GPU=2
    for std in 0. 0.5 1 2 4
    do 
        screen -dmS det$D$std$radius$run bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python comparison.py --D $D --WG 64 --radius $std --std 0.3 --scale 1";
    done
done







