import torch
import math
from copy import deepcopy
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import yolo

class YOLOloss(nn.Module):
    def __init__(self,anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):
        super(YOLOloss,self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing
        self.threshold = 4

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.cuda = cuda


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    pass