#!/bin/bash
is_test=$1
version=$2
stage=$3
caffe_root=/home/huijun/caffe-mod/
model_log_path=/home/huijun/DenseShuffleNet/logs/se-dpshufflenet_v${version}_${stage}.log
learning_curve_path=/home/huijun/DenseShuffleNet/logs/se-dpshufflenet_v${version}_${stage}.png

cd myscripts
python plot_learning_curve.py \
	${caffe_root} \
	${model_log_path} \
	${learning_curve_path} \
	${is_test}