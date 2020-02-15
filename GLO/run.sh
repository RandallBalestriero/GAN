#!/bin/bash
GPU=0
for Zstar in 5
    do
    for D in 16
    do
        for Z in 1 2 3 4 5 6 7
        do
            for W in 6
            do
                for N in 100 120 140 160 180 200 300 400 500 1000
                do
                    GPU=$(((GPU+1)%3))
                    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES='';python comparison.py --Z $Z --Zstar $Zstar --N $N --D $D --W $W";
                done
            done
        done
    done
done



