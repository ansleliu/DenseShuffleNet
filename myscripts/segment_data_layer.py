import os
import sys
caffe_root = "/home/huijun/caffe-mod/"
sys.path.insert(0, caffe_root + "python")
import caffe
from caffe.proto import caffe_pb2
import lmdb
import yaml
import numpy as np
import cStringIO as StringIO
from PIL import Image
from multiprocessing import Process, Queue


class SegmentDataLayer(caffe.Layer):
    """
        Segmentation data layer used for training.
    """

    def setup(self, bottom, top):
        """
            Setup the SegmentDataLayer.
        """
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._source = layer_params['source']
        self._mean_B = layer_params['mean_B']
        self._mean_G = layer_params['mean_G']
        self._mean_R = layer_params['mean_R']

        self._batch_size = layer_params['batch_size']
        self._crop_w = layer_params['crop_w']
        self._crop_h = layer_params['crop_h']

        self._img_mean = np.array((self._mean_B, self._mean_G, self._mean_R))

        self.set_roidb()  # Load lmdb for training

        top[0].reshape(self._batch_size, 3, self._crop_h, self._crop_w)  # Reshape blob data
        top[1].reshape(self._batch_size, 1, self._crop_h, self._crop_w)  # Reshape blob label

    def forward(self, bottom, top):
        """
            Get blobs and copy them into this layer's top blob vector.
        """
        blobs = self._get_next_minibatch()  # Get a blob

        for blob_id, blob in enumerate(blobs):   # blob[0] for image, blob[1] for mask
            # Reshape net's input blobs
            top[blob_id].reshape(*(blob.shape))  # Reshape top tensor/blob
            top[blob_id].data[...] = blob.astype(np.float32, copy=False)  # Feed corresponding data into top tensor/blob

    def backward(self, top, propagate_down, bottom):
        """
            This layer does not propagate gradients.
        """
        pass

    def reshape(self, bottom, top):
        """
            Reshaping happens during the call to forward.
        """
        pass

    def set_roidb(self):
        """
            Set the roidb to be used by this layer during training.
        """
        lmdb_dir = "{}".format(self._source)

        data_lmdb_path = os.path.join(lmdb_dir, "data")
        mask_lmdb_path = os.path.join(lmdb_dir, "label")

        self._data_lmdb = lmdb.open(data_lmdb_path, readonly=True)
        self._mask_lmdb = lmdb.open(mask_lmdb_path, readonly=True)

        self._txn = [self._data_lmdb.begin(), self._mask_lmdb.begin()]

        self._blob_queue = Queue(20)
        self._prefetch_process = BlobFetcher(self._blob_queue, self._txn, self._img_mean, self._crop_w, self._crop_h)
        self._prefetch_process.start()

        # Terminate the child process when the parent exists
        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()

            # close all db
            self._data_lmdb.close()
            self._mask_lmdb.close()

        import atexit
        atexit.register(cleanup)

    def _get_next_minibatch(self):
        """
            Return the blobs to be used for the next minibatch.
            The blobs will be computed in a separate process
            and made available through self._blob_queue.
        """
        images = np.zeros((self._batch_size, 3, self._crop_h, self._crop_w), dtype=np.float32)
        masks = np.zeros((self._batch_size, 1, self._crop_h, self._crop_w), dtype=np.float32)

        shuffled_batch = np.arange(self._batch_size)
        np.random.shuffle(shuffled_batch)
        for batch_index in shuffled_batch:
            blob_queue = self._blob_queue.get()
            images[batch_index, :, :, :] = blob_queue[0]
            masks[batch_index, :, :, :] = blob_queue[1]

        return [images, masks]


class BlobFetcher(Process):
    """
        Experimental class for prefetching blobs in a separate process.
    """
    def __init__(self, queue, txn, img_mean, crop_w, crop_h):
        super(BlobFetcher, self).__init__()

        self._queue = queue
        self._txn = txn
        self._cursor = [self._txn[0].cursor(), self._txn[1].cursor()]

        self._img_mean = img_mean
        self._scale = 0.00390625

        self._crop_w = crop_w
        self._crop_h = crop_h

        self._data_datum = caffe_pb2.Datum()  # For image
        self._mask_datum = caffe_pb2.Datum()  # For mask

    def _get_next_minibatch_inds(self):
        """
            Return the roidb indices for the next minibatch.
        """
        img_next = self._cursor[0].next()
        msk_next = self._cursor[1].next()
        if img_next and msk_next:
            pass
        else:
            print 'BlobFetcher to begin because of cursor point to end.'
            self._cursor = [self._txn[0].cursor(), self._txn[1].cursor()]
            self._cursor[0].next()
            self._cursor[1].next()

    def run(self):
        print 'BlobFetcher started'
        while True:
            self._get_next_minibatch_inds()
            data = self._cursor[0].value()
            label = self._cursor[1].value()

            # +++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 1. Image
            # +++++++++++++++++++++++++++++++++++++++++++++++++ #
            self._data_datum.ParseFromString(data)
            stream = StringIO.StringIO(self._data_datum.data)
            org_img = Image.open(stream)
            org_img = np.array(org_img, dtype=np.uint8)
            image = org_img[..., ::-1].copy()

            image_h, image_w, _ = org_img.shape

            # ++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 1.1 Crop Image
            # ++++++++++++++++++++++++++++++++++++++++++++++++++ #
            crop_h = self._crop_h if (self._crop_h < image_h) else image_h
            crop_w = self._crop_w if (self._crop_w < image_w) else image_w

            diff_h = image_h - self._crop_h
            diff_w = image_w - self._crop_w
            start_x = np.random.randint(0, diff_w + 1, size=1)[0]
            start_y = np.random.randint(0, diff_h + 1, size=1)[0]
            end_x = start_x + crop_w
            end_y = start_y + crop_h

            image = image[start_y:end_y, start_x:end_x]

            image = image.astype(np.float32)
            image -= self._img_mean
            image *= self._scale

            image = np.transpose(image, (2, 0, 1))

            # +++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2. Mask
            # +++++++++++++++++++++++++++++++++++++++++++++++++ #
            self._mask_datum.ParseFromString(label)
            stream = StringIO.StringIO(self._mask_datum.data)
            mask = Image.open(stream)
            mask = np.array(mask, dtype=np.uint8)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.1 Crop Mask
            # ++++++++++++++++++++++++++++++++++++++++++++++++++ #
            mask = mask[start_y:end_y, start_x:end_x]
            mask = mask.astype(np.float32)
            mask = mask[np.newaxis, :, :]

            blobs = [image, mask]
            self._queue.put(blobs)
