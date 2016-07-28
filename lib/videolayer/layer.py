# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from config import cfg
from caffe.io import Transformer
import numpy as np
import lmdb
from multiprocessing import Process, Queue

class VideoDatumLayer(caffe.Layer):
    """Video Datum layer used for training."""


    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        return self._blob_queue.get()

    def set_vdb(self, vdb):
        """Set the roidb to be used by this layer during training."""
        self._vdb = vdb
        self._blob_queue = Queue(10)
        self._prefetch_process = BlobFetcher(self._blob_queue,self._vdb)
        self._prefetch_process.start()
        # Terminate the child process when the parent exists
        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()
        import atexit
        atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the VideoDatumLayer."""

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape((cfg.TRAIN.BATCH_SIZE, 3, cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH, 3 if cfg.TRAIN.MODE == 'RGB' \
            else 2 * cfg.TRAIN.L))
        self._name_to_top_map['data'] = idx
        idx += 1
        top[idx].reshape(1)
        self._name_to_top_map['label'] = idx
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, vdb):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._vdb = vdb
        self._curv = 0
        self._curd = 0
        self._cursor =[]
        for i in xrange(len(self._curd)):
            db=lmdb.open(self._vdb[i]['dname'],readonly=True,lock=False)
            txn=db.begin()
            self._cursor.append(txn.cursor())
            self._cursor[i].first()
        # fix the random seed for reproducibility

    def _update_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code

        if self._curv>=len(self._vdb[self._curd]['vnum'])-1:
            self._cursor[self._curd].seek_range(0)
            self._curd = self._curd+1 if self._curd < len(self._vdb)-1 else 0
            self._curv=0
        else:
            self._curv+=1

    def _get_minibatch_v(cursor, position):
        """ get video datum."""
        frames_per_video = cfg.TRAIN.FRAMES_PER_VIDEO
        batch_size = cfg.TRAIN.BATCH_SIZE
        # rate=frames_per_video/batch_size
        index = np.arange(1, frames_per_video / batch_size, frames_per_video).astype('int')
        fnames = ['video%08d_%04d' % (position, sind) for sind in index]
        assert(len(fnames)==batch_size)
        datum = caffe_pb2.Datum()
        shape = (batch_size, 3, cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH, 3 if cfg.TRAIN.MODE == 'RGB' \
            else 2 * cfg.TRAIN.L)
        blob = {'data': np.zeros(shape, dtype=np.float32), 'label': np.zeros((batch_size,), dtype=np.float32)}
        count = 0
        for i in np.arange(frames_per_video):
            if cursor.key() not in fnames:
                cursor.next()
            else:
                labels = np.zeros((0,), dtype=np.float32)
                shape[0] = 0
                data = np.zeros(shape, dtype = np.float32)
                for ii in xrange(cfg.TRAIN.L):
                    datum.ParseFromString(cursor.value())
                    labels = np.vstack((labels, datum.label.astype(np.float32)))
                    tdata = caffe.io.datum_to_array(datum)
                    data = np.hstack((data, tdata.astype(np.float32)))
                    cursor.next()
                for ii in xrange(cfg.TRAIN.L):
                    cursor.prev()
                blob['data'][count, ...] = data
                blob['label'][count] = labels[0]
                count+=1
        assert(batch_size==count)
        return blob

    def run(self):
        print 'BlobFetcher started'
        blob=self._get_minibatch_v(self._cursor[self._curd],self._curv)
        self._update_inds()
        self._queue.put(blob)
