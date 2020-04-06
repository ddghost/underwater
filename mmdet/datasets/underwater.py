import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class Underwater(CocoDataset):

    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')

    def load_annotations(self, ann_file):
        print(ann_file)
        self.coco = COCO(ann_file)
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
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        return super(Underwater, self)._filter_imgs(min_size)

    def _parse_ann_info(self, img_info, ann_info):
        return super(Underwater, self)._parse_ann_info(img_info, ann_info)

    def format_results(self, results, **kwargs):
        return super(Underwater, self).format_results(results, **kwargs)

    def prepare_test_img(self, idx):
        return super(self).prepare_test_img(idx)