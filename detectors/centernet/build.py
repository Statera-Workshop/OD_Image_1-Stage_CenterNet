"""
CenterNet ("Objects as Points") builder, ResNet-18 + deformable up-convolutions (arch resdcn_18), Image task only.

Paper: Zhou, Wang, Krahenbuhl, "Objects as Points" (arXiv 1904.07850, 2019).
Code : official xingyizhou/CenterNet src/lib
       models/networks/resnet_dcn.py  PoseResNet (BasicBlock ResNet-18, 3 x [DCN 3x3 + BN + ReLU + ConvTranspose
                                      4x4 s2 (bilinear init) + BN + ReLU] -> stride 4, heads hm / wh / reg with
                                      head_conv 64, hm bias -2.19)
       models/losses.py               FocalLoss (_neg_loss, CornerNet), RegL1Loss
       trains/ctdet.py                CtdetLoss: hm 1, wh 0.1, off 1
       datasets/sample/ctdet.py       targets: gaussian_radius(min_overlap 0.7), draw_umich_gaussian, ind, reg
       models/decode.py               _nms: 3x3 max-pool peak keep, box = centre + offset +- wh / 2, x down_ratio 4
       DCNv2 (CharlesShang, compiled CUDA) -> torchvision.ops.deform_conv2d with a modulation mask ([Unified]:
       the same modulated deformable convolution; DCNv2 initialisation kept: offsets / mask conv = 0,
       weight U(+-1/sqrt(n)), bias 0).

Raw contract: every output-stride-4 cell is a candidate (608 -> 152 x 152 = 23,104); class probabilities are the
heat map after the official 3x3 peak NMS (non-peaks -> 0), objectness = 1. The common post-processing then
applies its usual NMS ([Unified]; the paper uses only the peak extraction, top-100).

[Unified] no ImageNet ResNet-18 weights, trainable BN, common init (nn.Conv2d / BN only: the DCN and the
bilinear ConvTranspose2d keep their official init, hm bias -2.19 kept), no mean / std, CenterNet's random
scale / shift / colour augmentation replaced by the common augmentation, 608 input, Adam 1e-4
(paper Adam 1.25e-4 x batch 114 / 128). PreciseBN like RetinaNet (BN backbone + deconvs, micro batch 2).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from detectors.anchor_free.common import HeadStats, split_targets, to_raw

BN_MOMENTUM = 0.1
DOWN_RATIO = 4


class DCN(nn.Module):
    """DCNv2 DCN (modulated deformable conv): offset / mask predicted by a zero-initialised 3x3 conv."""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                 deformable_groups=1):
        super().__init__()
        self.stride, self.padding, self.dilation = stride, padding, dilation
        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        n = in_channels * kh * kw
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        # conv_offset_mask: kept as raw parameters so the common Conv2d init does not replace the zero init
        self.offset_weight = nn.Parameter(torch.zeros(deformable_groups * 3 * kh * kw, in_channels, kh, kw))
        self.offset_bias = nn.Parameter(torch.zeros(deformable_groups * 3 * kh * kw))

    def forward(self, input):
        out = F.conv2d(input, self.offset_weight, self.offset_bias, self.stride, self.padding, self.dilation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, mask=mask)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        super().__init__()
        self.inplanes = 64
        self.heads = heads
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(3, [256, 128, 64], [4, 4, 4])
        for head, classes in self.heads.items():
            fc = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True),
                               nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True))
            if "hm" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = num_kernels[i], 1, 0          # _get_deconv_cfg(4)
            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(planes, planes, kernel_size=kernel, stride=2, padding=padding,
                                    output_padding=output_padding, bias=False)
            fill_up_weights(up)
            layers += [fc, nn.BatchNorm2d(planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
                       up, nn.BatchNorm2d(planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)]
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.deconv_layers(x)
        return {head: self.__getattr__(head)(x) for head in self.heads}


# ---- utils/image.py (targets) -----------------------------------------------------------------------------------
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# ---- models/losses.py --------------------------------------------------------------------------------------------
def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def _neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds).sum()
    num_pos = pos_inds.float().sum()
    return -neg_loss if num_pos == 0 else -(pos_loss + neg_loss) / num_pos


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return feat.gather(1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2)))


def reg_l1_loss(output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / (mask.sum() + 1e-4)


class CenterNet(nn.Module):
    def __init__(self, num_classes, hm_weight=1.0, wh_weight=0.1, off_weight=1.0, max_objs=128):
        super().__init__()
        self.num_classes = num_classes
        self.net = PoseResNet(BasicBlock, [2, 2, 2, 2], {"hm": num_classes, "wh": 2, "reg": 2}, head_conv=64)
        self.hm_weight, self.wh_weight, self.off_weight = hm_weight, wh_weight, off_weight
        self.max_objs = max_objs
        self.precise_bn = True                 # [Unified] common/precise_bn.py
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    @torch.no_grad()
    def build_targets(self, targets, out_h, out_w, device):
        """datasets/sample/ctdet.py on input-pixel XYXY boxes (augmentation already applied by the common loader)."""
        B = len(targets)
        hm = np.zeros((B, self.num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((B, self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((B, self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((B, self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((B, self.max_objs), dtype=np.uint8)
        for b, t in enumerate(targets):
            boxes = (t["boxes"].cpu().numpy() / DOWN_RATIO).copy()
            labels = t["labels"].cpu().numpy()
            for k in range(min(len(labels), self.max_objs)):
                bbox, cls_id = boxes[k], int(labels[k])
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, out_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, out_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm[b, cls_id], ct_int, radius)
                    wh[b, k] = 1. * w, 1. * h
                    ind[b, k] = ct_int[1] * out_w + ct_int[0]
                    reg[b, k] = ct - ct_int
                    reg_mask[b, k] = 1
        to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
        return to(hm), to(wh), to(reg), to(ind), to(reg_mask)

    def forward(self, x, targets=None):
        out = self.net(x)
        B, _, H, W = out["hm"].shape
        heat = out["hm"].sigmoid()
        hmax = F.max_pool2d(heat, (3, 3), stride=1, padding=1)                  # decode._nms
        heat_nms = heat * (hmax == heat).float()
        ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=heat.dtype),
                                torch.arange(W, device=x.device, dtype=heat.dtype), indexing="ij")
        reg, wh = out["reg"], out["wh"]
        cx, cy = xs + reg[:, 0], ys + reg[:, 1]
        boxes = torch.stack([cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2], -1)
        boxes = boxes.reshape(B, -1, 4) * DOWN_RATIO
        raw = to_raw(boxes, heat_nms.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
        if targets is None:
            return raw
        tgt = split_targets(targets, B, x.shape[2])
        hm_t, wh_t, reg_t, ind_t, mask_t = self.build_targets(tgt, H, W, x.device)
        hm_loss = _neg_loss(_sigmoid(out["hm"].clone()), hm_t)
        wh_loss = reg_l1_loss(out["wh"], mask_t, ind_t, wh_t)
        off_loss = reg_l1_loss(out["reg"], mask_t, ind_t, reg_t)
        loss_dict = {"hm_loss": self.hm_weight * hm_loss, "wh_loss": self.wh_weight * wh_loss,
                     "off_loss": self.off_weight * off_loss}
        loss_dict["losses"] = sum(loss_dict.values())
        loss_dict["num_pos"] = mask_t.float().sum() / B
        cell_label = torch.full((B, H * W), -1, dtype=torch.long, device=x.device)   # class of each centre cell
        for b, t in enumerate(tgt):
            n = min(len(t["labels"]), self.max_objs)
            keep = mask_t[b, :n].bool()
            cell_label[b, ind_t[b, :n][keep]] = t["labels"][:n][keep]
        flat_pos = (cell_label >= 0).view(-1)
        fg = cell_label.view(-1)[flat_pos]
        logits = out["hm"].detach().permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        self.stats.update(loss_dict, logits.sigmoid().max(-1).values, logits, flat_pos, fg)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("CenterNet has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return CenterNet(num_classes)
