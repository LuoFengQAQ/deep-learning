import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import nms


def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_class):
    # ..............................................#
    # input: bitch_size,3 * (5+num_classes) , 20 ,20
    # where 3代表每一个特征点上3个先验框，5=4+1：4为先验框的回归系数，1代表先验框内是否有物体
    # num_classes代表物体种类，20*20代表特征图的长宽
    # ..............................................#
    batch_size = input.size(0)
    input_height = input.size(2)
    input_width = input.size(3)

    # ..............................................#
    # input_shape为输入图片的尺寸
    # 及640*640
    # 则stride_h= stride_w =32
    # 表示32个像素点被处理为一个特征点
    # ..............................................#

    stride_h = input_shape[0] / input_height
    stride_w = input_shape[1] / input_width

    # ----------------------------------------------#
    # 特征层的先验框
    # 对先验框的宽高进行缩放
    # ----------------------------------------------#
    scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                      anchors[anchors_mask[2]]]

    # ----------------------------------------------#
    # 对网络的输出进行reshape
    # input: bitch_size,3 * (5+num_classes) , 20 ,20
    # output: batch_size ,3 ,20, 20, 5+num_classes
    # ----------------------------------------------#
    prediction = input.view(batch_size, len(anchors_mask[2]),  # 特征层先验框的数量，理论上为3
                            num_class + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

    # ----------------------------------------------#
    # 先验框中心位置参数调整
    # ----------------------------------------------#
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    # ----------------------------------------------#
    # 先验框宽高调整参数
    # ----------------------------------------------#
    w = torch.sigmoid(prediction[..., 2])
    h = torch.sigmoid(prediction[..., 3])
    # ----------------------------------------------#
    # 获得置信度，是否有物体 0-1
    # ----------------------------------------------#
    conf = torch.sigmoid(prediction[..., 4])
    # ----------------------------------------------#
    # 种类置信度
    # ----------------------------------------------#
    pred_cls = torch.sigmoid(prediction[..., 5:])
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    Longtensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    # ----------------------------------------------#
    # 生成网络，先验框中心，网格左上角
    # batch_size，3，20，20
    # range(20)
    # [
    #  [0,1,2,...,19],
    #  [0,1,2,...,19],
    #  ...20列
    #  [0,1,2,...,19]
    #  batch_size,3,20,20
    # ]
    # ----------------------------------------------#
    grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
        batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
    grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).repeat(
        batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)
    # ----------------------------------------------#
    # 按照网格格式生成先验框的宽高
    # batch_size,3,20,20
    # ----------------------------------------------#

    anchor_w = FloatTensor(scaled_anchors).index_select(1, Longtensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, Longtensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
    # ----------------------------------------------#
    # 利用预测结果对先验框进行调整
    # 首先调整先验框的中心，从先验框中心向右下角偏移
    # 再调整先验框的宽高
    # ----------------------------------------------#
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data * 2 - 0.5 + grid_x
    pred_boxes[..., 1] = y.data * 2 - 0.5 + grid_y
    pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
    pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h
