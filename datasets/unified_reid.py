# encoding: utf-8
"""
Unified ReID Dataset for SOLIDER-REID.
Combines DukeMTMC-reID, Market-1501, IUST, CUHK03, and VNPT1.

Camera-ID offset strategy (to avoid global collisions):
  - DukeMTMC-reID : cams 1-8   → offset 0   → global cams  0-7
  - Market-1501   : cams 1-6   → offset 8   → global cams  8-13
  - IUST          : cams 1-32* → offset 14  → global cams 14-30 (*non-contiguous, remapped)
  - CUHK03        : cams 1-2   → offset 31  → global cams 31-32
  - VNPT1         : cams 1-11  → offset 33  → global cams 33-43

PID offsets are accumulated dynamically based on selected subsets.

Subset selection:
  Pass `subsets` (list[str]) to select which sub-datasets to include.
  Valid names: 'duke', 'market', 'iust', 'cuhk03', 'vnpt1'.
  None / [] → use ALL sub-datasets.
  Example: subsets=['duke', 'vnpt1']
"""

import glob
import re
import os.path as osp

from .bases import BaseImageDataset

# ── Global camera-ID offsets ───────────────────────────────────────────────────
DUKE_CAM_OFFSET   = 0   # Duke  : 0-7
MARKET_CAM_OFFSET = 8   # Market: 8-13
IUST_CAM_OFFSET   = 14  # IUST  : 14-30 (17 non-contiguous cams, remapped 0-16, +14)
CUHK03_CAM_OFFSET = 31  # CUHK03: 31-32
VNPT1_CAM_OFFSET  = 33  # VNPT1 : 33-43

ALL_SUBSETS = ('duke', 'market', 'iust', 'cuhk03', 'vnpt1')


class UnifiedReID(BaseImageDataset):
    """
    Unified ReID Dataset combining up to 5 sub-datasets.

    Args:
        root (str): Root directory containing ``Unified-ReID-Dataset/``.
        verbose (bool): Print dataset statistics.
        pid_begin (int): Starting PID number.
        subsets (list[str] | None): Subset names to include. None → ALL.
    """

    dataset_dir = 'Unified-ReID-Dataset'

    DUKE_DIR   = 'dukememc-reid'
    MARKET_DIR = 'market1501'
    IUST_DIR   = 'iust'
    CUHK03_DIR = 'cuhk03'
    VNPT1_DIR  = 'vnpt1'

    def __init__(self, root='', verbose=True, pid_begin=0, subsets=None, **kwargs):
        super(UnifiedReID, self).__init__()

        self.root        = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.pid_begin   = pid_begin

        # ── Resolve subsets ────────────────────────────────────────────────
        if not subsets:
            self.subsets = list(ALL_SUBSETS)
        else:
            for s in subsets:
                if s not in ALL_SUBSETS:
                    raise ValueError(f"Unknown subset '{s}'. Valid: {ALL_SUBSETS}")
            self.subsets = list(subsets)

        # ── Build paths ────────────────────────────────────────────────────
        def _paths(sub_dir):
            d = osp.join(self.dataset_dir, sub_dir)
            return (osp.join(d, 'bounding_box_train'),
                    osp.join(d, 'query'),
                    osp.join(d, 'bounding_box_test'))

        self.duke_train,   self.duke_query,   self.duke_gallery   = _paths(self.DUKE_DIR)
        self.market_train, self.market_query, self.market_gallery = _paths(self.MARKET_DIR)
        self.iust_train,   self.iust_query,   self.iust_gallery   = _paths(self.IUST_DIR)
        self.cuhk03_train, self.cuhk03_query, self.cuhk03_gallery = _paths(self.CUHK03_DIR)
        self.vnpt1_train,  self.vnpt1_query,  self.vnpt1_gallery  = _paths(self.VNPT1_DIR)

        self._check_before_run()

        # ── Registry ───────────────────────────────────────────────────────
        sub_registry = {
            'duke':   (self._process_duke,   self.duke_train),
            'market': (self._process_market, self.market_train),
            'iust':   (self._process_iust,   self.iust_train),
            'cuhk03': (self._process_cuhk03, self.cuhk03_train),
            'vnpt1':  (self._process_vnpt1,  self.vnpt1_train),
        }

        # ── Merge train splits ─────────────────────────────────────────────
        train = []
        running_pid_offset = pid_begin
        subset_info = []

        for name in self.subsets:
            process_fn, train_dir = sub_registry[name]
            raw_data, n_pids = process_fn(train_dir, relabel=True)
            train.extend(self._apply_pid_offset(raw_data, running_pid_offset))
            subset_info.append((name, n_pids, self._cam_offset(name)))
            running_pid_offset += n_pids

        # ── Eval: use Market-1501 query/gallery ────────────────────────────
        query   = self._process_market(self.market_query,   relabel=False)[0]
        gallery = self._process_market(self.market_gallery, relabel=False)[0]

        if verbose:
            print(f"=> UnifiedReID loaded  (subsets: {self.subsets})")
            for name, n_pids, cam_off in subset_info:
                print(f"   {name:<8s}  train_pids: {n_pids:<5d}  cam_offset: {cam_off}")
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _cam_offset(name):
        return {'duke': DUKE_CAM_OFFSET, 'market': MARKET_CAM_OFFSET,
                'iust': IUST_CAM_OFFSET, 'cuhk03': CUHK03_CAM_OFFSET,
                'vnpt1': VNPT1_CAM_OFFSET}[name]

    @staticmethod
    def _apply_pid_offset(dataset, offset):
        return [(p, pid + offset, camid, trackid) for p, pid, camid, trackid in dataset]

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"Dataset root not found: '{self.dataset_dir}'")
        subset_dirs = {
            'duke':   [self.duke_train,   self.duke_query,   self.duke_gallery],
            'market': [self.market_train, self.market_query, self.market_gallery],
            'iust':   [self.iust_train,   self.iust_query,   self.iust_gallery],
            'cuhk03': [self.cuhk03_train, self.cuhk03_query, self.cuhk03_gallery],
            'vnpt1':  [self.vnpt1_train,  self.vnpt1_query,  self.vnpt1_gallery],
        }
        for name in self.subsets:
            for d in subset_dirs[name]:
                if not osp.exists(d):
                    raise RuntimeError(f"Required path not found for '{name}': '{d}'")
        # Market always needed for eval
        for d in [self.market_query, self.market_gallery]:
            if not osp.exists(d):
                raise RuntimeError(f"Market eval path not found: '{d}'")

    # ── Per-dataset processors ─────────────────────────────────────────────────

    def _process_duke(self, dir_path, relabel=False):
        """Format: 0001_c2_f0046182.jpg"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        pid_container = set()
        for p in img_paths:
            pid, _ = map(int, pattern.search(p).groups())
            pid_container.add(pid)
        pid2label = {pid: lbl for lbl, pid in enumerate(sorted(pid_container))}
        dataset = []
        for p in sorted(img_paths):
            pid, camid = map(int, pattern.search(p).groups())
            camid = (camid - 1) + DUKE_CAM_OFFSET
            if relabel:
                pid = pid2label[pid]
            dataset.append((p, pid, camid, 1))
        return dataset, len(pid_container)

    def _process_market(self, dir_path, relabel=False):
        """Format: 0001_c1s1_000151_01.jpg"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        pid_container = set()
        for p in sorted(img_paths):
            pid, _ = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: lbl for lbl, pid in enumerate(sorted(pid_container))}
        dataset = []
        for p in sorted(img_paths):
            pid, camid = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            camid = (camid - 1) + MARKET_CAM_OFFSET
            if relabel:
                pid = pid2label[pid]
            dataset.append((p, pid, camid, 1))
        return dataset, len(pid_container)

    def _process_iust(self, dir_path, relabel=False):
        """Format: 0001_c2s1_024962_01.jpg  (Market-like, non-contiguous cam IDs)"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        pid_container = set()
        raw_cams = set()
        for p in img_paths:
            pid, camid = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
            raw_cams.add(camid)
        pid2label = {pid: lbl for lbl, pid in enumerate(sorted(pid_container))}
        cam2idx   = {cam: idx for idx, cam in enumerate(sorted(raw_cams))}
        dataset = []
        for p in sorted(img_paths):
            pid, camid = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            camid = cam2idx[camid] + IUST_CAM_OFFSET
            if relabel:
                pid = pid2label[pid]
            dataset.append((p, pid, camid, 1))
        return dataset, len(pid_container)

    def _process_cuhk03(self, dir_path, relabel=False):
        """Format: 0001_c1_5.png"""
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'(\d{4})_c(\d+)_\d+\.png')
        pid_container = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m:
                pid_container.add(int(m.group(1)))
        pid2label = {pid: lbl for lbl, pid in enumerate(sorted(pid_container))}
        dataset = []
        for p in sorted(img_paths):
            m = pattern.search(osp.basename(p))
            if not m:
                continue
            pid   = int(m.group(1))
            camid = (int(m.group(2)) - 1) + CUHK03_CAM_OFFSET
            if relabel:
                pid = pid2label[pid]
            dataset.append((p, pid, camid, 1))
        return dataset, len(pid_container)

    def _process_vnpt1(self, dir_path, relabel=False):
        """Format: 0001_c3s1_1772593372001_00.jpg  (Market-like)"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        pid_container = set()
        for p in img_paths:
            pid, _ = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: lbl for lbl, pid in enumerate(sorted(pid_container))}
        dataset = []
        for p in sorted(img_paths):
            pid, camid = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            camid = (camid - 1) + VNPT1_CAM_OFFSET
            if relabel:
                pid = pid2label[pid]
            dataset.append((p, pid, camid, 1))
        return dataset, len(pid_container)
