#!/bin/bash
gpu_id=$1

caffe_root=/home/huijun/caffe-mod
solver_path=/home/huijun/DenseShuffleNet/models/SE-DPShuffleNet_MVD_solver.prototxt

# weights=/home/huijun/DenseShuffleNet/weights/se-dpshufflenet-mvd_iter_1000.caffemodel --weights=${weights}
# snapshot=/home/huijun/DenseShuffleNet/weights/denseshufflenet_v1_coco_iter_4000.solverstate

log_path=/home/huijun/DenseShuffleNet/logs/se-dpshufflenet_mvd.log

${caffe_root}/build/tools/caffe train --solver=${solver_path} --gpu=${gpu_id} 2>&1 | tee ${log_path}
