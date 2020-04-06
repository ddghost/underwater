import numpy as np
from pycocotools.coco import COCO
import random
import math
from .underwater import Underwater
from .registry import DATASETS


@DATASETS.register_module
class UnderwaterSample():
    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')
    def __init__(self, ann_file, **kwags):
        self.UnderwaterBig = Underwater(ann_file[0], **kwags)
        self.UnderwaterSmall = Underwater(ann_file[1], **kwags)



    def get_ann_info(self, idx):
        p = 0.2
        if(random.random() < p):
            return self.UnderwaterBig.get_ann_info(idx)

        else:
            return self.UnderwaterSmall.get_ann_info(idx)


