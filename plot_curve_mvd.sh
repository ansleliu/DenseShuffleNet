#!/bin/bash
is_test=0
caffe_root=/home/huijun/caffe-mod/
model_log_path=/home/huijun/DenseShuffleNet/logs/se-dpshufflenet_mvd.log
learning_curve_path=/home/huijun/DenseShuffleNet/logs/se-dpshufflenet_mvd.png

cd myscripts
python plot_learning_curve.py \
	${caffe_root} \
	${model_log_path} \
	${learning_curve_path} \
	${is_test}