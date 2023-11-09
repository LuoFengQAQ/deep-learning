import torch


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

    x = torch