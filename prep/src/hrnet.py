"""
HRNet-W32 backbone matching mmpose 0.x AP-10K checkpoint keys exactly.

Key layout in checkpoint:
  backbone.conv1 / bn1 / conv2 / bn2          — stem (stride-2 twice → H/4)
  backbone.layer1.{0-3}.*                      — 4 Bottleneck blocks, 64→256 ch
  backbone.transition1.{0,1}.*                 — split 256 → [32, 64]
  backbone.stage2.0.branches.{0,1}.{0-3}.*    — 2-branch HRModule, BasicBlock×4
  backbone.stage2.0.fuse_layers.*
  backbone.transition2.{2}.*                  — add 128-ch branch
  backbone.stage3.{0-3}.*                     — 4× HRModule, 3 branches
  backbone.transition3.{3}.*                  — add 256-ch branch
  backbone.stage4.{0-2}.*                     — 3× HRModule, 4 branches
  keypoint_head.final_layer.weight/bias        — (17, 32, 1, 1)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels,
                 multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.num_inchannels = list(num_inchannels)
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        in_ch = self.num_inchannels[branch_index]
        out_ch = num_channels[branch_index]
        if stride != 1 or in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
        layers = [block(in_ch, out_ch, stride, downsample)]
        self.num_inchannels[branch_index] = out_ch * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], out_ch))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            out_c = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], out_c, 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(out_c),
                            ))
                        else:
                            out_c = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], out_c, 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = None
            for j in range(self.num_branches):
                f = fuse_layer[j]
                feat = x[j] if f is None else f(x[j])
                y = feat if y is None else y + feat
            x_fuse.append(self.relu(y))
        return x_fuse


def _make_layer(block, inplanes, planes, num_blocks):
    downsample = None
    if inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, 1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, downsample=downsample)]
    in_ch = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(in_ch, planes))
    return nn.Sequential(*layers)


def _make_transition(num_channels_pre, num_channels_cur):
    num_branches_pre = len(num_channels_pre)
    num_branches_cur = len(num_channels_cur)
    transition_layers = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur[i] != num_channels_pre[i]:
                transition_layers.append(nn.Sequential(
                    nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                    nn.BatchNorm2d(num_channels_cur[i]),
                    nn.ReLU(inplace=True),
                ))
            else:
                transition_layers.append(None)
        else:
            conv3x3s = []
            for j in range(i + 1 - num_branches_pre):
                in_c = num_channels_pre[-1]
                out_c = num_channels_cur[i] if j == i - num_branches_pre else in_c
                conv3x3s.append(nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ))
            transition_layers.append(nn.Sequential(*conv3x3s))
    return nn.ModuleList(transition_layers)


def _make_stage(num_modules, num_branches, block, num_blocks, num_channels,
                num_inchannels, multi_scale_output=True):
    modules = []
    for i in range(num_modules):
        mso = multi_scale_output or i < num_modules - 1
        m = HighResolutionModule(
            num_branches, block, num_blocks, num_inchannels, num_channels, mso)
        num_inchannels = m.num_inchannels
        modules.append(m)
    return nn.Sequential(*modules), num_inchannels


class HRNetW32(nn.Module):
    """
    HRNet-W32 backbone reproducing mmpose 0.x key names exactly.
    forward() returns the highest-resolution branch: (N, 32, H/4, W/4).
    """

    def __init__(self):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ── Layer 1: 4 Bottleneck, 64 → 256 ch ───────────────────────────────
        self.layer1 = _make_layer(Bottleneck, 64, 64, 4)

        # ── Stage 2: 2 branches [32, 64] ─────────────────────────────────────
        ch2 = [32, 64]
        self.transition1 = _make_transition([256], ch2)
        self.stage2, pre2 = _make_stage(
            num_modules=1, num_branches=2, block=BasicBlock,
            num_blocks=[4, 4], num_channels=ch2, num_inchannels=list(ch2),
            multi_scale_output=True)

        # ── Stage 3: 3 branches [32, 64, 128] ────────────────────────────────
        ch3 = [32, 64, 128]
        self.transition2 = _make_transition(pre2, ch3)
        self.stage3, pre3 = _make_stage(
            num_modules=4, num_branches=3, block=BasicBlock,
            num_blocks=[4, 4, 4], num_channels=ch3, num_inchannels=list(ch3),
            multi_scale_output=True)

        # ── Stage 4: 4 branches [32, 64, 128, 256] ───────────────────────────
        ch4 = [32, 64, 128, 256]
        self.transition3 = _make_transition(pre3, ch4)
        self.stage4, _ = _make_stage(
            num_modules=3, num_branches=4, block=BasicBlock,
            num_blocks=[4, 4, 4, 4], num_channels=ch4, num_inchannels=list(ch4),
            multi_scale_output=False)

    @staticmethod
    def _apply_transition(x_list, transition):
        out = []
        for i, t in enumerate(transition):
            src = x_list[i] if i < len(x_list) else x_list[-1]
            out.append(src if t is None else t(src))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        x = self.layer1(x)

        x_list = self._apply_transition([x], self.transition1)
        x_list = self.stage2(x_list)

        x_list = self._apply_transition(x_list, self.transition2)
        x_list = self.stage3(x_list)

        x_list = self._apply_transition(x_list, self.transition3)
        x_list = self.stage4(x_list)

        return x_list[0]  # 32 ch at H/4 × W/4
