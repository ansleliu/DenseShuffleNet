import cv2
import sys
caffe_root = "/home/huijun/caffe-mod/"
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import cv2
from PIL import Image, ImageEnhance

gpu_id = int(0)
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

# caffe.set_mode_cpu()

caffe_net = caffe.Net("/home/huijun/DenseShuffleNet/models/SE-DPShuffleNet_V1_deploy.prototxt",
                      "/home/huijun/DenseShuffleNet/weights/se-dpshufflenet-v1_iter_7000.caffemodel",
                      caffe.TEST)

# net.forward()
# for each layer, show the output shape
# for layer_name, blob in caffe_net.blobs.iteritems():
#     print layer_name + '\t' + str(blob.data.shape)


image_mean = np.array((80.5423, 91.3162, 81.4312))
image_name = "munster_000168_000019_leftImg8bit"
image_org = cv2.imread("/home/huijun/DenseShuffleNet/deploy/{}.png".format(image_name))

"""
pil_image = Image.fromarray(image_org)
enhancer = ImageEnhance.Color(pil_image)
pil_image = enhancer.enhance(0.75)
# enhancer = ImageEnhance.Contrast(pil_image)
# pil_image = enhancer.enhance(1.25)
enhancer = ImageEnhance.Brightness(pil_image)
pil_image = enhancer.enhance(1.25)
enhancer = ImageEnhance.Sharpness(pil_image)
pil_image = enhancer.enhance(1.25)
image_org = np.array(pil_image)
image_org = image_org[:, :, ::-1].copy()
"""

image = cv2.resize(image_org, (1024, 512), cv2.INTER_CUBIC)
image = image.astype(np.float32)
image -= image_mean
image *= 0.00390625
image = np.transpose(image, (2, 0, 1))
image = image[np.newaxis, :, :, :]      # (B, C, H, W)

caffe_net.blobs['data'].data[...] = image
infos = caffe_net.forward()

prediction = infos['prob'][0]
prediction = prediction.argmax(axis=0).astype(np.uint8)
prediction = np.squeeze(prediction)

pil_image = Image.fromarray(prediction)
pil_image = pil_image.resize((2048, 1024))
prediction = np.array(pil_image)
prediction = np.resize(prediction, (3, 1024, 2048))

prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

label_colours = cv2.imread("/home/huijun/DenseShuffleNet/myscripts/cityscapes_colormap.png").astype(np.uint8)
prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
label_colours_bgr = label_colours[..., ::-1]
cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
img_msk = cv2.addWeighted(image_org, 0.25, prediction_rgb, 0.75, 0)

save_mask_name = "/home/huijun/DenseShuffleNet/output/{}_mask.png".format(image_name)
save_imgmsk_name = "/home/huijun/DenseShuffleNet/output/{}_imgmsk.png".format(image_name)
cv2.imwrite(save_mask_name, prediction_rgb)
cv2.imwrite(save_imgmsk_name, img_msk)

mask = cv2.imread("/home/huijun/DenseShuffleNet/deploy/munster_000168_000019_gtFine_color.png")
mask = cv2.resize(mask, (1024, 512), cv2.INTER_CUBIC)

cv2.namedWindow("Org Mask", cv2.WINDOW_NORMAL)
cv2.imshow('Org Mask', mask)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.imshow('Mask', prediction_rgb)
cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
cv2.imshow('Segmentation', img_msk)
cv2.waitKey(0)

print "Ok!"
