#!/bin/bash

GPU=0
for RUN in 6
do
#    GPU=$(((GPU+1)%2))
#    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i mnist.py --LR 0.0001 --Z 10 --MODEL CONVVAE --RUN $RUN";
#    GPU=$(((GPU+1)%2))
#    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python  -i mnist.py --LR 0.0003 --Z 10 --MODEL CONVGAN --RUN $RUN";
    GPU=$(((GPU+1)%2))
    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python mnist.py --LR 0.0001 --Z 10 --MODEL BETACONVVAE --RUN $RUN";

#    GPU=$(((GPU+1)%2))
#    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i mnist.py --LR 0.0001 --Z 10 --MODEL VAE --RUN $RUN";
#    GPU=$(((GPU+1)%2))
#    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python  -i mnist.py --LR 0.0003 --Z 10 --MODEL GAN --RUN $RUN";
    GPU=$(((GPU+1)%2))
    screen -dmS ab bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python mnist.py --LR 0.0001 --Z 10 --MODEL BETAVAE --RUN $RUN";
done



