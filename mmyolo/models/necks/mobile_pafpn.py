# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import List, Union, Type, Optional
#
# import torch.nn as nn
# from mmdet.utils import ConfigType, OptMultiConfig
#
# from mmyolo.registry import MODELS
# # from .. import UIB_Down
# from ..utils import make_divisible, make_round
# from .yolov5_pafpn import YOLOv5PAFPN
# import torch
# from mmcv.cnn import ConvModule
# from timm.models._efficientnet_blocks import MultiSpectralAttentionLayer, SqueezeExcite, OrthoAttentionLayer
#
# ModuleType = Type[nn.Module]
#
# @MODELS.register_module()
# class MobilePAFPN(YOLOv5PAFPN):
#     """Path Aggregation Network used in YOLOv8.
#
#     Args:
#         in_channels (List[int]): Number of input channels per scale.
#         out_channels (int): Number of output channels (used at each scale)
#         deepen_factor (float): Depth multiplier, multiply number of
#             blocks in CSP layer by this amount. Defaults to 1.0.
#         widen_factor (float): Width multiplier, multiply number of
#             channels in each layer by this amount. Defaults to 1.0.
#         num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
#         freeze_all(bool): Whether to freeze the model
#         norm_cfg (dict): Config dict for normalization layer.
#             Defaults to dict(type='BN', momentum=0.03, eps=0.001).
#         act_cfg (dict): Config dict for activation layer.
#             Defaults to dict(type='SiLU', inplace=True).
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Defaults to None.
#     """
#
#     def __init__(self,
#                  in_channels: List[int],
#                  out_channels: Union[List[int], int],
#                  deepen_factor: float = 1.0,
#                  widen_factor: float = 1.0,
#                  num_csp_blocks: int = 3,
#                  freeze_all: bool = False,
#                  norm_cfg: ConfigType = dict(
#                      type='BN', momentum=0.03, eps=0.001),
#                  act_cfg: ConfigType = dict(type='SiLU', inplace=True),
#                  init_cfg: OptMultiConfig = None,
#                  se_layer: Optional[ModuleType] = None
#                  ):
#         self.se_layer = se_layer
#         super().__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             deepen_factor=deepen_factor,
#             widen_factor=widen_factor,
#             num_csp_blocks=num_csp_blocks,
#             freeze_all=freeze_all,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#             init_cfg=init_cfg)
#
#     def build_reduce_layer(self, idx: int) -> nn.Module:
#         """build reduce layer.
#
#         Args:
#             idx (int): layer idx.
#
#         Returns:
#             nn.Module: The reduce layer.
#         """
#         reduce = nn.Identity()
#         # if self.se_layer == 'MultiSpectralAttentionLayer':
#         #     reduce = MultiSpectralAttentionLayer(self.in_channels[idx], 16, 16)
#         # if self.se_layer == 'OrthoAttentionLayer':
#         #     reduce = OrthoAttentionLayer(self.in_channels[idx])
#
#         return reduce
#
#     def build_top_down_layer(self, idx: int) -> nn.Module:
#         """build top down layer.
#
#         Args:
#             idx (int): layer idx.
#
#         Returns:
#             nn.Module: The top down layer.
#         """
#
#
#         return UIB_Down(
#             make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
#                            1),
#             make_divisible(self.out_channels[idx - 1], 1),
#             se_layer = self.se_layer
#             )
#
#     def build_bottom_up_layer(self, idx: int) -> nn.Module:
#         """build bottom up layer.
#
#         Args:
#             idx (int): layer idx.
#
#         Returns:
#             nn.Module: The bottom up layer.
#         """
#
#         return UIB_Down(
#             make_divisible(
#                 (self.out_channels[idx] + self.out_channels[idx + 1]),
#                 1),
#             make_divisible(self.out_channels[idx + 1], 1),
#             se_layer = self.se_layer
#             )
#
#     def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
#         """build upsample layer."""
#         return nn.Upsample(scale_factor=2, mode='nearest')
#
#     def build_downsample_layer(self, idx: int) -> nn.Module:
#         """build downsample layer.
#
#         Args:
#             idx (int): layer idx.
#
#         Returns:
#             nn.Module: The downsample layer.
#         """
#         return ConvModule(
#             make_divisible(self.in_channels[idx], 1),
#             make_divisible(self.in_channels[idx], 1),
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#
#     def forward(self, inputs: List[torch.Tensor]) -> tuple:
#
#
#         # print('input0:',inputs[0].shape)
#         # print('input1:',inputs[1].shape)
#         # print('input2:',inputs[2].shape)
#
#         """Forward function."""
#         assert len(inputs) == len(self.in_channels)
#         # reduce layers
#         reduce_outs = []
#         for idx in range(len(self.in_channels)):
#             reduce_outs.append(self.reduce_layers[idx](inputs[idx]))
#
#         # top-down path
#         inner_outs = [reduce_outs[-1]]
#         for idx in range(len(self.in_channels) - 1, 0, -1):
#             feat_high = inner_outs[0]
#             feat_low = reduce_outs[idx - 1]
#             upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
#                                                  idx](
#                                                      feat_high)
#             if self.upsample_feats_cat_first:
#                 top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
#             else:
#                 top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
#             inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
#                 top_down_layer_inputs)
#             inner_outs.insert(0, inner_out)
#
#         # bottom-up path
#         outs = [inner_outs[0]]
#         for idx in range(len(self.in_channels) - 1):
#             feat_low = outs[-1]
#             feat_high = inner_outs[idx + 1]
#             downsample_feat = self.downsample_layers[idx](feat_low)
#             out = self.bottom_up_layers[idx](
#                 torch.cat([downsample_feat, feat_high], 1))
#             outs.append(out)
#
#         # out_layers
#         results = []
#         for idx in range(len(self.in_channels)):
#             results.append(self.out_layers[idx](outs[idx]))
#
#         return tuple(results)