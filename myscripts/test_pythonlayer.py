import sys
import cv2
caffe_root = "/home/huijun/caffe-mod/"
sys.path.insert(0, caffe_root + "python")
import caffe
import numpy as np
from PIL import Image, ImageEnhance

gpu_id = int(0)
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

caffe_net = caffe.Net("/home/huijun/DenseShuffleNet/models/pdn.prototxt",
                      caffe.TEST)
for layer_name, blob in caffe_net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


image_mean = np.array((80.5423, 91.3162, 81.4312))
image_name = "munster_000168_000019_leftImg8bit"
image_org = cv2.imread("/home/huijun/DenseShuffleNet/deploy/{}.png".format(image_name))

image = cv2.resize(image_org, (768, 384), cv2.INTER_CUBIC)
image = image.astype(np.float32)
image -= image_mean
image *= 0.00390625
image = np.transpose(image, (2, 0, 1))
image = image[np.newaxis, :, :, :]      # (B, C, H, W)

caffe_net.blobs['data'].data[...] = image
infos = caffe_net.forward()

print "Ok!"
