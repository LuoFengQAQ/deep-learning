import torch
from torch import nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s, p, g, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), p, g, act)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1

            )
        )


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        if self.add:
            x = x + self.cv2(self.cv1(x))
        else:
            x = self.cv2(self.cv1(x))
        return x


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_,c_,shortcut,g,e=1.0)for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.m(x1)
        x2 = self.cv2(x)
        x = torch.cat(
            (x1, x2), dim=1
        )
        return x


class SPP(nn.Module):
    def __init__(self, c1, c2,k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k)+1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x,stride=1,padding=x//2)for x in k]
        )

    def forward(self,x):
        x = self.cv1(x)
        x = torch.cat(
            [x, self.m[1](x), self.m[2](x), self.m[3](x)], dim=1
        )
        return self.cv2(x)


class CSPdarknet(nn.Module):
    def __init__(self,base_channels,base_depth,phi,pretrained):
        super().__init__()
        # ........................ #
        # 输入层图像为3*640*640
        # base_channels初始为64
        # base_depth残差网络层数
        # ........................ #

        self.stem = Focus(3,base_channels)
        # .......................... #
        # 第二层输出为128
        # 图像尺寸为160*160
        # .......................... #
        self.dark2 = torch.nn.Sequential(
            Conv(base_channels, base_channels*2, 3, 2),
            C3(base_channels*2,base_channels*2,base_depth),# 该层不对图像的尺寸进行改变
        )
        # .......................... #
        # 第三层输出为256
        # 图像尺寸为80*80
        # .......................... #
        self.dark3 = nn.Sequential(
            Conv(base_channels*2,base_channels*4,3,2),
            C3(base_channels*4,base_channels*4,base_depth*3)
        )
        # .......................... #
        # 第四层输出为512
        # 图像尺寸为40*40
        # .......................... #
        self.dark4 = nn.Sequential(
            Conv(base_channels*4,base_channels*8,3,2),
            C3(base_channels*8,base_channels*8,base_depth*3)
        )
        # ----------------------------------------------- #
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # ----------------------------------------------- #
        self.dark5 = nn.Sequential(
            Conv(base_channels*8,base_channels*16,3,2),
            SPP(base_channels*16,base_channels*16),
            C3(base_channels*16,base_channels*16,base_depth,shortcut=False)
        )
        if pretrained:
            url = {
                's' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_s_backbone.pth',
                'm' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_m_backbone.pth',
                'l' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_l_backbone.pth',
                'x' : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_x_backbone.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from ", url.split('/')[-1])

    def forward(self,x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x

        return feat1,feat2,feat3



