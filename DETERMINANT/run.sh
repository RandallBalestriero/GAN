#!/bin/bash
GPU=0
for run in 0 1 2 3 4 5
do
    for D in 2 4 6 8 10 12
    do
        GPU=$(((GPU+1)%3))
        for std in 0.033 0.066 0.1 0.133
        do 
            for radius in 0.5 1 1.5 2
            do
                screen -dmS det$D$std$radius$run bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python comparison.py --D $D --WG 64 --n_modes 4 --radius $radius --std $std --run $run";
            done
        done
    done
    sleep 2h
done



