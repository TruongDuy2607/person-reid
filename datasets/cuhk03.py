# encoding: utf-8
"""
CUHK03 dataset for SOLIDER-REID.
Filename format: 0001_c1_5.png  →  pid=1, camid=1
Cameras: 1-2
"""

import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class CUHK03(BaseImageDataset):
    """
    CUHK03 Re-ID dataset.
    Filename format: {pid:04d}_c{camid}_{index}.png
    Dataset statistics:
        # train identities: 767
        # cameras: 2
        # images: 7368 (train) + 1400 (query) + 5328 (gallery)
    """
    dataset_dir = 'cuhk03'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(CUHK03, self).__init__()
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
            print("=> CUHK03 loaded")
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
        # Format: 0001_c1_5.png
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'(\d{4})_c(\d+)_\d+\.png')

        pid_container = set()
        for img_path in img_paths:
            m = pattern.search(osp.basename(img_path))
            if m is None:
                continue
            pid_container.add(int(m.group(1)))
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            m = pattern.search(osp.basename(img_path))
            if m is None:
                continue
            pid   = int(m.group(1))
            camid = int(m.group(2)) - 1  # 0-based
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
