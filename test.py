import math

import torch
import torch.nn as nn
from torchsummary import summary

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# INSN模块
# class ELANs(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c1 * e)  # hidden channels 256
#         self.cv1 = Conv(c1, c_, 1, 1)   # 512 256
#         self.cv2 = DWConv(c_, c_, 3, 1)  # 256 256
#         self.cv3 = Conv(c_, c2, 1, 1)  # 256 256
#         self.add = shortcut and c_ == c2
#
#     def forward(self, x):   # 1*512*40*40
#         x = self.cv1(x)     # 1*256*40*40
#         out = self.cv3(self.cv2(x))     # 1*256*40*40
#         return out + x if self.add else out     # 1*256*40*40
#
# class INSNs(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
#         super().__init__()
#         c_ = int(c1 * e)
#         self.conv1 = Conv(c1, c_, 1, 1)     # 512 256
#         self.conv2_1 = Conv(c_, c_, 1, 1)   # 256 256
#         self.conv2_2 = DWConv(c_, c_, 3, 1)   # 256 256
#         self.conv3 = Conv(c1, c2, 1, 1)     # 512 256
#         self.elan = nn.Sequential(*(ELANs(c1, c_, shortcut, e=0.5) for _ in range(n))) # 512 256
#
#     def forward(self, x):  # 1*512*40*40
#         x1 = self.elan(x)   # 1*256*40*40
#         x2 = self.conv1(x)  # 1*256*40*40
#         x2_1 = self.conv2_1(x2)     # 1*256*40*40
#         x2_2 = self.conv2_2(x2)     # 1*256*40*40
#         x3 = x2 + x2_1 + x2_2       # 1*256*40*40
#         out = self.conv3(torch.cat((x1, x3), dim=1))    # 1*256*40*40
#         return out

# class ELANs(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c1 * e)  # hidden channels 256
#         self.cv1 = Conv(c1, c_, 1, 1)   # 512 256
#         self.cv2 = DWConv(c_, c_, 3, 1)  # 256 256
#         self.cv3 = Conv(c_, c_, 1, 1)  # 256 256
#         self.cv4 = Conv(int(c_ / 0.5), c2, 1, 1)
#         self.add = shortcut and c_ == c2
#
#     def forward(self, x):   # 1*512*40*40
#         x1 = self.cv1(x)     # 1*256*40*40
#         out = self.cv4(torch.cat((x1, self.cv3(self.cv2(x1))),dim=1))
#         return out + x if self.add else out     # 1*256*40*40
#
#
# class INSNs(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, e=0.25): # 512 512
#         super().__init__()
#         c_ = int(c1 * e)
#         self.conv1 = Conv(c1, c_, 1, 1)     # 512 256
#         self.conv2_1 = Conv(c_, c_, 1, 1)   # 256 256
#         self.conv2_2 = DWConv(c_, c_, 3, 1)   # 256 256
#         self.conv3 = Conv(c_, c_, 1, 1)     # 512 256
#         self.elan = nn.Sequential(*(ELANs(c1, c_, shortcut, e=e) for _ in range(n))) # 512 256
#         self.conv4 = Conv(int(c_ / 0.5), c2, 1, 1)
#
#     def forward(self, x):  # 1*512*40*40
#         x1 = self.elan(x)   # 1*256*40*40
#         x2 = self.conv1(x)  # 1*256*40*40
#         x2_1 = self.conv2_1(x2)     # 1*256*40*40
#         x2_2 = self.conv2_2(x2)     # 1*256*40*40
#         x3 = self.conv3(x2 + x2_1 + x2_2)      # 1*256*40*40
#         out = self.conv4(torch.cat((x1, x3), dim=1))    # 1*256*40*40
#         return out

# PMSA
# class ELANl(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = DWConv(c_, c_, 3, 1)
#         self.cv3 = Conv(c_, c_, 1, 1, g=g)
#         self.cv4 = DWConv(c_, c_, 3, 1)
#         self.cv5 = Conv(c_, c_, 1, 1, g=g)
#         self.cv6 = Conv(c_ * 3, c2, 1, 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = self.cv3(self.cv2(x1))
#         x3 = self.cv5(self.cv4(x2))
#         out = torch.cat((x1, x2, x3), dim=1)
#         return self.cv6(out) + x1 if self.add else self.cv6(out)
#
#
# class INSNl(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c1 * e)
#         self.conv1 = Conv(c1, c_, 1, 1)
#         self.conv2_1 = DWConv(c_, c_, 3, 1)
#         self.conv2_2 = DWConv(c_, c_, 5, 1)
#         self.conv2_3 = DWConv(c_, c_, 7, 1)
#         self.conv3 = Conv(c_, c_, 1, 1)
#         self.conv4 = Conv(c1, c2, 1, 1)
#         self.elan = nn.Sequential(*(ELANl(c1, c_, shortcut, g, e=e) for _ in range(n)))
#
#     def forward(self, x):  # 1*64*160*160
#         x1 = self.elan(x)
#         x2 = self.conv1(x)
#         x2_1 = self.conv2_1(x2)
#         x2_2 = self.conv2_2(x2)
#         x2_3 = self.conv2_3(x2)
#         x3 = x2 + x2_1 + x2_2 + x2_3
#         x3 = self.conv3(x3)
#         out = self.conv4(torch.cat((x1, x3), dim=1))
#         return out

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class INSNl(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.25):
        super().__init__()
        c_ = int(c1 * e )
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = DWConv(c_, c_, 3, 1)
        self.conv3 = DWConv(c_, c_, 3, 1)
        self.conv4 = DWConv(c_, c_, 3, 1)
        self.conv5 = DWConv(c_, c_, 3, 1)
        self.conv6 = DWConv(c_, c_, 3, 1)
        self.conv7 = DWConv(c_, c_, 3, 1)
        self.conv8 = DWConv(c_, c_, 1, 1)
        self.conv9 = DWConv(c_, c_, 1, 1)
        self.conv10 = DWConv(c_, c_, 1, 1)
        self.conv11 = DWConv(c_, c_, 1, 1)
        self.conv12 = Conv(c1, c2, 1, 1)

    def forward(self, x):  # 1*64*160*160
        x1 = self.conv1(x)
        x2 = self.conv3(self.conv2(x1))
        x3 = self.conv5(self.conv4(x2))
        x4 = self.conv6(x1)
        x5 = self.conv8(self.conv7(x1))
        x6 = self.conv11(self.conv10(self.conv9(x1)))
        out = self.conv12(torch.cat((x1, x2, x3, x1 + x4 + x5 + x6), dim=1))
        return out

if __name__ == '__main__':

    # INSNl
    channel, w_h = 64, 160
    model = DW(channel, channel, 1).cuda()
    input_data = torch.randn(1, channel, w_h, w_h).cuda()

    # INSNs
    # channel, w_h = 512, 20
    # model = INSNs(channel, int(channel), 1).cuda()
    # input_data = torch.randn(1, channel, w_h, w_h).cuda()


    # 前向传播
    output = model(input_data)
    print(output.shape)
    # 查看模型
    print(model)
    # 计算参数量
    summary(model, input_size=(channel, w_h, w_h))


