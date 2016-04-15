#!/bin/bash

python train.py

tensorboard tensorboard --logdir=logs
