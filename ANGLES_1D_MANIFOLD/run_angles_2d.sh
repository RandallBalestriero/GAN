#!/bin/bash
GPU=0
for D in 6 8
do
    GPU=$(((GPU+1)%3))
    for run in 0 1 2 3 4 5 6 7 8 9
    do 
        screen -dmS angle$D$run bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python angles_circle.py --WG $D --run $run";
    done
done



