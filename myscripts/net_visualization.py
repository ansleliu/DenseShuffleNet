import os
import cv2
import sys
caffe_root = "/home/huijun/caffe-mod/"
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from matplotlib import pyplot as plt

# mean_value channel [0]: 87.6562
# mean_value channel [1]: 93.561
# mean_value channel [2]: 96.9831


def fit_size(images, input_height, input_width):
    h, w, _ = images[0].shape
    if h == input_height and w == input_width:
        return images

    h_ratio = input_height * 1.0 / h
    w_ratio = input_width * 1.0 / w

    ratio = min(h_ratio, w_ratio)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    new_imgs = []
    # print('{:d},{:d} --> {:d},{:d}'.format(h, w, new_h, new_w))

    for img in images:
        if len(img.shape) == 2:
            new_img = np.zeros((input_height, input_width), np.uint8)
        else:
            new_img = np.zeros((input_height, input_width, 3), np.uint8)
        new_img[:] = 0
        h_offset = (input_height - new_h) / 2
        w_offset = (input_width - new_w) / 2
        new_img[h_offset:h_offset + new_h, w_offset:w_offset + new_w] = cv2.resize(img, (new_w, new_h))
        new_imgs.append(new_img)
    return new_imgs


def convert(data):
    # data = fit_size([data, ], in_height, in_width)[0]
    data = cv2.resize(data, (in_width, in_height), interpolation=cv2.INTER_CUBIC)
    # Read mean image

    data = data.astype(np.float32)
    data -= img_mean
    data *= 0.00390625

    data = np.transpose(data, (2, 0, 1))
    data = data[np.newaxis, :, :, :]
    return data


def vis_square(data, layer):
    """
    Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    cv2.namedWindow('{} viz'.format(layer), cv2.WINDOW_NORMAL)
    cv2.imshow('{} viz'.format(layer), data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("conv5_1_viz.png", data)
    # plt.imshow(data, cmap='brg', interpolation='bicubic')
    # plt.imshow(data)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.axis('off')
    # plt.show()
    plt.imsave("/home/huijun/DenseShuffleNet/netviz/" + layer + ".png", data)

if __name__ == "__main__":
    gpu_id = int(0)
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    # caffe.set_mode_cpu()

    model_def = "/home/huijun/DenseShuffleNet/models/SE-DPShuffleNet_V1_deploy.prototxt"
    model_weights = "/home/huijun/DenseShuffleNet/weights/se-dpshufflenet-v1_iter_12000.caffemodel"

    det_net = caffe.Net(model_def,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)

    data_root = "/home/huijun/DenseShuffleNet/deploy"
    image_name = "munster_000168_000019_leftImg8bit.png"
    img_path = os.path.join(data_root, image_name)
    image = cv2.imread(img_path)

    in_height = 512
    in_width = 1024
    img_mean = np.array((80.5423, 91.3162, 81.4312))

    img_crop = cv2.resize(image, (in_width, in_height), interpolation=cv2.INTER_CUBIC)
    img_in = img_crop.astype(np.float32)
    img_in -= img_mean
    img_in *= 0.00390625
    img_in = np.transpose(img_in, (2, 0, 1))
    img_in = img_in[np.newaxis, :, :, :]

    # transformed_image = trans.preprocess('data', image)
    # plt.imshow(image)

    det_net.blobs['data'].data[...] = img_in

    # perform classification
    output = det_net.forward()

    layer_names = ["data_conv1", "data_conv1_maxout", "data_conv2", "data_conv2_maxout",
                    "decode1_aconv1_interp", "decode1_aconv3_1", "decode1_aconv3_2", "decode1_aconv3_3",
                   "decode2_aconv1_interp", "decode2_aconv3_1", "decode2_aconv3_2", "decode2_aconv3_3",
                   "decode3_aconv1_interp", "decode3_aconv3_1", "decode3_aconv3_2", "decode3_aconv3_3",
                   "decode1_score", "decode2_score",
                   "decode3_score", "decode_score_sum",
                   "decode_score"]

    for _, layer in enumerate(layer_names):
        # the parameters are a list of [weights, biases]
        # filters = caffe_net.params['conv3_1'][0].data
        # vis_square(np.reshape(filters, (32*64, 5, 5)))

        print "> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <"
        print "> Processing layer {}".format(layer)
        feat = det_net.blobs[layer].data[0][:9]
        vis_square(feat, layer)

    print "> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <"
    print "> Done !!!"
    print "> ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <"
    cv2.waitKey(0)
    cv2.destroyAllWindows()
