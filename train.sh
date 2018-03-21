#!/bin/bash
gpu_id=$1
version=$2
stage=$3

caffe_root=/home/huijun/caffe-mod
solver_path=/home/huijun/DenseShuffleNet/models/SE-DPShuffleNet_V${version}_solver.prototxt

weights=/home/huijun/DenseShuffleNet/weights/se-dpshufflenet-mvd_iter_56140.caffemodel
# snapshot=/home/huijun/DenseShuffleNet/weights/se-dpshufflenet-v1_iter_17000.solverstate

log_path=/home/huijun/DenseShuffleNet/logs/se-dpshufflenet_v${version}_${stage}.log

${caffe_root}/build/tools/caffe train --solver=${solver_path} --weights=${weights} --gpu=${gpu_id} 2>&1 | tee ${log_path}
