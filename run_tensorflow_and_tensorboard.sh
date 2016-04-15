#!/bin/bash

#Remove previous logs
rm logs/t*/events.out*
# Run tensorflow
python train.py
# Run Tensorboard
tensorboard tensorboard --logdir=logs
