#!/bin/sh
pip install gensim
pip install s3fs
/usr/bin/python3 train_multi_label.py --batch_size 128 --epochs 1 --learning_rate 0.01 --model_dir /opt/ml/model