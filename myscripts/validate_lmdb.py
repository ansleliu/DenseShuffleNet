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

            mask = mask[start_y:end_y, start_x:end_x]
            mask = mask.astype(np.float32)
            mask = mask[np.newaxis, :, :]

            blobs = [image, mask]
            self._queue.put(blobs)


if __name__ == "__main__":
    lmdb_dir = "/media/datavolume3/huijun/CityScapes/cityscapes_lmdb_train"
    image_mean = np.array((83.8269, 93.2754, 84.0849))
    crop_h, crop_w = 384, 768

    data_lmdb_path = os.path.join(lmdb_dir, "data")
    mask_lmdb_path = os.path.join(lmdb_dir, "label")

    data_lmdb = lmdb.open(data_lmdb_path, readonly=True)
    mask_lmdb = lmdb.open(mask_lmdb_path, readonly=True)

    txn = [data_lmdb.begin(), mask_lmdb.begin()]

    blob_queue = Queue(20)
    prefetch_process = BlobFetcher(blob_queue, txn, image_mean, crop_w, crop_h)
    prefetch_process.start()

    # Terminate the child process when the parent exists
    def cleanup():
        print 'Terminating BlobFetcher'
        prefetch_process.terminate()
        prefetch_process.join()

        # close all db
        data_lmdb.close()
        mask_lmdb.close()


    import atexit

    atexit.register(cleanup)

    while True:
        images = np.zeros((5, 3, crop_h, crop_w), dtype=np.float32)
        masks = np.zeros((5, 1, crop_h, crop_w), dtype=np.float32)

        shuffled_batch = np.arange(5)
        np.random.shuffle(shuffled_batch)
        for batch_index in shuffled_batch:
            blobs = blob_queue.get()
            images[batch_index, :, :, :] = blobs[0]
            masks[batch_index, :, :, :] = blobs[1]
