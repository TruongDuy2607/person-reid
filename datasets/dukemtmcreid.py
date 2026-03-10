# encoding: utf-8
"""
DukeMTMC-reID dataset for SOLIDER-REID.
Filename format: 0001_c2_f0046182.jpg  →  pid=1, camid=2
"""

import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class DukeMTMCreID(BaseImageDataset):
    """
    DukeMTMC-reID
    Reference:
        Ristani et al. Performance Measures and a Data Set for Multi-Target,
        Multi-Camera Tracking. ECCVW 2016.
    Dataset statistics:
        # train identities: 702
        # images: 16522 (train) + 2228 (query) + 17661 (gallery)
        # cameras: 8
    """
    dataset_dir = 'dukememc-reid'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(DukeMTMCreID, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir   = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir   = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train   = self._process_dir(self.train_dir,   relabel=True)
        query   = self._process_dir(self.query_dir,   relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for d in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

    def _process_dir(self, dir_path, relabel=False):
        # Format: 0001_c2_f0046182.jpg
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            camid -= 1  # 0-based
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
