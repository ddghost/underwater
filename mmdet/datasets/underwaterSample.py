import numpy as np
from pycocotools.coco import COCO
import random
import math
from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class UnderwaterSample(CocoDataset):

    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')

    def load_annotations(self, ann_file):
        #ann_file(big,small)
        self.coco = COCO(ann_file[0])
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        self.cocoAnother = COCO(ann_file[1])
        self.cat_idsAnother = self.cocoAnother.getCatIds()
        self.cat2labelAnother = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_idsAnother)
        }
        self.img_idsAnother = self.cocoAnother.getImgIds()
        self.img_infosAnother = []
        for i in selfAnother.img_ids:
            info = selfAnother.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infosAnother.append(info)
        
        return img_infos




    def get_ann_info(self, idx):
        p = 0.8
        if(random.random() < p):
            img_info = self.img_infos[idx]
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
        else:
            if(idx >= len(self.img_infosAnother) ):
                 idx = random.random() * self.img_infosAnother
                 idx = math.floor(idx)
            img_info = self.img_infosAnother[idx]
            img_id = img_info['id']
            ann_ids = self.cocoAnother.getAnnIds(imgIds=[img_id])
            ann_info = self.cocoAnother.loadAnns(ann_ids)

        return self._parse_ann_info(img_info, ann_info)

    def _filter_imgs(self, min_size=32):
        return super(self)._filter_imgs(min_size)

    def _parse_ann_info(self, img_info, ann_info):
        return super(self)._parse_ann_info(img_info, ann_info)

    def format_results(self, results, **kwargs):
        return super(self).format_results(results, **kwargs)

    def prepare_test_img(self, idx):
        return super(self).prepare_test_img(idx)