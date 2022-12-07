#!/bin/bash

BUCKET_START=$1
BUCKET_END=$2

# git pull
python3 train_range.py $BUCKET_START $BUCKET_END --hflip 0.5 --random_crop_pad 4 --crop --cutout --rand_aug 14 --dont_save "${@:3}"
