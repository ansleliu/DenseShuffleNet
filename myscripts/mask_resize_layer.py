import os
import sys
import cv2
caffe_root = "/home/huijun/yolo_seg_caffe/caffe-yolov2/"
sys.path.insert(0, caffe_root + "python")
import caffe
import yaml
import numpy as np


class MaskResizeLayer(caffe.Layer):
    """
        YoLoSeg mask resize layer used for training.
    """

    def setup(self, bottom, top):
        """
            Setup the MaskResizeLayer.
        """
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._new_c = layer_params['new_C']
        self._new_w = layer_params['new_W']
        self._new_h = layer_params['new_H']

        # Batch size must be 1, here set 1 as default value, not load form config file
        top[0].reshape(1, self._new_c, self._new_h, self._new_w)   # Reshape tensor size

    def forward(self, bottom, top):
        """
            Get blobs and copy them into this layer's top blob vector.
        """
        batch_size = bottom[0].num

        mask_c = bottom[0].channels
        if mask_c != 1:
            raise Exception("The Channel of the mask must be one.")

        # ---------------------------------------------------- #
        # 1. Bottom[0] and Top[0]
        # ---------------------------------------------------- #
        bottom_blob = bottom[0].data  # Target

        blob = np.zeros((batch_size, self._new_c, self._new_h, self._new_w))
        for batch_idx in np.arange(batch_size):
            # ------------------------------------------------ #
            # 1. Resize Targets
            # ------------------------------------------------ #
            bottom_temp = bottom_blob[batch_idx, :, :, :]
            bottom_temp = np.transpose(bottom_temp, (1, 2, 0))
            bottom_temp = cv2.resize(bottom_temp, (self._new_w, self._new_h), cv2.INTER_CUBIC)
            bottom_temp = bottom_temp[:, :, np.newaxis]
            bottom_temp = np.transpose(bottom_temp, (2, 0, 1))

            # ------------------------------------------------ #
            # 2. Copy Targets
            # ------------------------------------------------ #
            for channel_idx in np.arange(self._new_c):
                blob[batch_idx, channel_idx, :, :] = bottom_temp

        top[0].reshape(*(blob.shape))  # Reshape top tensor/blob
        top[0].data[...] = blob.astype(np.float32, copy=False)  # Feed corresponding data into top tensor/blob

    def backward(self, top, propagate_down, bottom):
        """
            This layer propagate gradients.
        """
        pass

    def reshape(self, bottom, top):
        """
            Reshaping happens during the call to forward.
        """
        pass
