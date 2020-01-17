#!/bin/bash
screen -dmS ab bash -c "python -i mnist.py --LR 0.0003";
screen -dmS ab bash -c "python -i mnist.py --LR 0.00029";
screen -dmS ab bash -c "python -i mnist.py --LR 0.00031";
screen -dmS ab bash -c "python -i mnist.py --LR 0.000312";
screen -dmS ab bash -c "python -i mnist.py --LR 0.000311";









