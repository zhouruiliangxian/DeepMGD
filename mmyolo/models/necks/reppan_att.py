# 2023.09.18-Changed for Neck implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn

from mmyolo.models.necks.common2 import RepVGGBlock, BottleRep, BepC3, RepBlock, SimConv
from timm.models.repvit import RepViTBlock
from mmyolo.models.necks.layers import Conv
from mmyolo.models.necks.common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
from mmyolo.models.necks.transformer import PyramidPoolAgg, TopBasicLayer, InjectionMultiSum_Auto_pool,Attention
from mmyolo.registry import MODELS
from mmyolo.models.modules import HGBlock,RepC3,GhostBottleneck,SPPF,C2f,C2fCIB

@MODELS.register_module()
class RepGD_att_Neck(nn.Module):
    def __init__(
            self,
            channels_list=None,
            num_repeats=None,
            # block=RepVGGBlock,
            block=HGBlock,
            extra_cfg=None
    ):
        super().__init__()
        # print(extra_cfg.fusion_in)
        if isinstance(block, str):
            block = eval(block)
        assert channels_list is not None
        assert num_repeats is not None
        
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]),
                     kernel_size=1, stride=1, padding=0),
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[4],  # 1024
                out_channels=channels_list[5],  # 512
                kernel_size=1,
                stride=1
        )
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
                out_channels=channels_list[5],  # 256
            mobile=True
        )
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5],
                                                     norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p4 = RepBlock(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[5],  # 256
                n=num_repeats[0],
                block=block
        )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[6],  # 128
                kernel_size=1,
                stride=1
        )
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channels_list[2], channels_list[2]],  # 512, 256
                out_channels=channels_list[6],  # 256
            mobile=True
        )
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[6], channels_list[6],
                                                     norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[6],  # 128
                out_channels=channels_list[6],  # 128
                n=num_repeats[1],
                block=block
        )
        
        self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
        # self.high_IFM = TopBasicLayer(
        #         block_num=extra_cfg.depths,
        #         embedding_dim=extra_cfg.embed_dim_n,
        #         key_dim=extra_cfg.key_dim,
        #         num_heads=extra_cfg.num_heads,
        #         mlp_ratio=extra_cfg.mlp_ratios,
        #         attn_ratio=extra_cfg.attn_ratios,
        #         drop=0, attn_drop=0,
        #         drop_path=dpr,
        #         norm_cfg=extra_cfg.norm_cfg
        # )
        # self.high_IFM = nn.Sequential(
        #         Conv(extra_cfg.embed_dim_n, extra_cfg.embed_dim_n, kernel_size=1, stride=1, padding=0),
        #         *[block(extra_cfg.embed_dim_n, extra_cfg.embed_dim_n) for _ in range(extra_cfg.fuse_block_num)],
        #         Conv(extra_cfg.embed_dim_n, extra_cfg.embed_dim_n,
        #              kernel_size=1, stride=1, padding=0),
        # )
        # self.high_IFM = RepViTBlock(
        #     extra_cfg.embed_dim_n, mlp_ratio=2, kernel_size=5, use_se=True, act_layer=nn.GELU
        # )
        self.high_IFM = nn.Sequential(
                Conv(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), kernel_size=1, stride=1, padding=0),
                *[RepViTBlock(sum(extra_cfg.trans_channels[2:4]), mlp_ratio=2, kernel_size=5, use_se=True, act_layer=nn.GELU) for _ in range(extra_cfg.fuse_block_num)],
                Conv(sum(extra_cfg.trans_channels[2:4]), sum(extra_cfg.trans_channels[2:4]),
                     kernel_size=1, stride=1, padding=0),
        )
        # self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
        
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[6] + channels_list[7],  # 128 + 128
                out_channels=channels_list[8],  # 256
                n=num_repeats[2],

                block=block
        )
        
        self.LAF_n5 = AdvPoolFusion()
        # self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
        #                                              norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6,
                                                     )

        self.Rep_n5 = Attention(
            dim=channels_list[5] + channels_list[9], key_dim=64, num_heads=8
        )
        
        self.trans_channels = extra_cfg.trans_channels

    def forward(self, input):
        (c2, c3, c4, c5) = input

        # print(f"Input shapes: c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}, c5: {c5.shape}")

        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        # print(f"low_align_feat shape: {low_align_feat.shape}")

        low_fuse_feat = self.low_IFM(low_align_feat)
        # print(f"low_fuse_feat shape: {low_fuse_feat.shape}")

        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        # print(f"low_global_info shapes: {[info.shape for info in low_global_info]}")

        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        # print(f"c5_half shape: {c5_half.shape}")

        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        # print(f"p4_adjacent_info shape: {p4_adjacent_info.shape}")

        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        # print(f"p4 shape after Inject_p4: {p4.shape}")

        p4 = self.Rep_p4(p4)
        # print(f"p4 shape after Rep_p4: {p4.shape}")

        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        # print(f"p4_half shape: {p4_half.shape}")

        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        # print(f"p3_adjacent_info shape: {p3_adjacent_info.shape}")

        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        # print(f"p3 shape after Inject_p3: {p3.shape}")

        p3 = self.Rep_p3(p3)
        # print(f"p3 shape after Rep_p3: {p3.shape}")

        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        # print(f"high_align_feat shape: {high_align_feat.shape}")

        high_fuse_feat = self.high_IFM(high_align_feat)
        # print(f"high_fuse_feat shape: {high_fuse_feat.shape}")

        # high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        # print(f"high_fuse_feat shape after conv_1x1_n: {high_fuse_feat.shape}")

        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        # print(f"high_global_info shapes: {[info.shape for info in high_global_info]}")

        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        # print(f"n4_adjacent_info shape: {n4_adjacent_info.shape}")

        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        # print(f"n4 shape after Inject_n4: {n4.shape}")

        n4 = self.Rep_n4(n4)
        # print(f"n4 shape after Rep_n4: {n4.shape}")

        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        # print(f"n5_adjacent_info shape: {n5_adjacent_info.shape}")

        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        # print(f"n5 shape after Inject_n5: {n5.shape}")

        n5 = self.Rep_n5(n5)
        # print(f"n5 shape after Rep_n5: {n5.shape}")

        outputs = [p3, n4, n5]

        return outputs

# @MODELS.register_module()
class RepGDNeck_repvit(nn.Module):
    def __init__(
            self,
            channels_list=None,
            num_repeats=None,
            block=RepViTBlock,
            extra_cfg=None
    ):
        super().__init__()
        # print(extra_cfg.fusion_in)
        assert channels_list is not None
        assert num_repeats is not None

        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
            Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
            *[block(extra_cfg.embed_dim_p,  mlp_ratio=2, kernel_size=3, use_se=True, act_layer=nn.GELU) for _ in range(extra_cfg.fuse_block_num)],
            Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]),
                 kernel_size=1, stride=1, padding=0),
        )

        self.reduce_layer_c5 = SimConv(
            in_channels=channels_list[4],  # 1024
            out_channels=channels_list[5],  # 512
            kernel_size=1,
            stride=1
        )
        self.LAF_p4 = SimFusion_3in(
            in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
            out_channels=channels_list[5],  # 256
            mobile=True
        )
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5],
                                                     norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p4 = nn.Sequential(*(block(channels_list[5],  mlp_ratio=2, kernel_size=3, use_se=True, act_layer=nn.GELU) for _ in range(num_repeats[0])))

        self.reduce_layer_p4 = SimConv(
            in_channels=channels_list[5],  # 256
            out_channels=channels_list[6],  # 128
            kernel_size=1,
            stride=1
        )
        self.LAF_p3 = SimFusion_3in(
            in_channel_list=[channels_list[2], channels_list[2]],  # 512, 256
            out_channels=channels_list[6],  # 256
            mobile=True
        )
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[6], channels_list[6],
                                                     norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p3 = nn.Sequential(*(block(channels_list[6],  mlp_ratio=2, kernel_size=3, use_se=True, act_layer=nn.GELU) for _ in range(num_repeats[1])))


        self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
        self.high_IFM = TopBasicLayer(
            block_num=extra_cfg.depths,
            embedding_dim=extra_cfg.embed_dim_n,
            key_dim=extra_cfg.key_dim,
            num_heads=extra_cfg.num_heads,
            mlp_ratio=extra_cfg.mlp_ratios,
            attn_ratio=extra_cfg.attn_ratios,
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=extra_cfg.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)

        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Rep_n4 = nn.Sequential(*(block(channels_list[6] + channels_list[7],  mlp_ratio=2, kernel_size=3, use_se=True, act_layer=nn.GELU) for _ in range(num_repeats[2])))


        self.LAF_n5 = AdvPoolFusion()
        # self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
        #                                              norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6,
                                                     )

        self.Rep_n5 = nn.Sequential(*(block(channels_list[5] + channels_list[9],  mlp_ratio=2, kernel_size=3, use_se=True, act_layer=nn.GELU) for _ in range(num_repeats[3])))


        self.trans_channels = extra_cfg.trans_channels

    def forward(self, input):
        (c2, c3, c4, c5) = input

        # print(f"Input shapes: c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}, c5: {c5.shape}")

        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        # print(f"low_align_feat shape: {low_align_feat.shape}")

        low_fuse_feat = self.low_IFM(low_align_feat)
        # print(f"low_fuse_feat shape: {low_fuse_feat.shape}")

        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        # print(f"low_global_info shapes: {[info.shape for info in low_global_info]}")

        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        # print(f"c5_half shape: {c5_half.shape}")

        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        # print(f"p4_adjacent_info shape: {p4_adjacent_info.shape}")

        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        # print(f"p4 shape after Inject_p4: {p4.shape}")

        p4 = self.Rep_p4(p4)
        # print(f"p4 shape after Rep_p4: {p4.shape}")

        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        # print(f"p4_half shape: {p4_half.shape}")

        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        # print(f"p3_adjacent_info shape: {p3_adjacent_info.shape}")

        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        # print(f"p3 shape after Inject_p3: {p3.shape}")

        p3 = self.Rep_p3(p3)
        # print(f"p3 shape after Rep_p3: {p3.shape}")

        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        # print(f"high_align_feat shape: {high_align_feat.shape}")

        high_fuse_feat = self.high_IFM(high_align_feat)
        # print(f"high_fuse_feat shape: {high_fuse_feat.shape}")

        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        # print(f"high_fuse_feat shape after conv_1x1_n: {high_fuse_feat.shape}")

        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        # print(f"high_global_info shapes: {[info.shape for info in high_global_info]}")

        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        # print(f"n4_adjacent_info shape: {n4_adjacent_info.shape}")

        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        # print(f"n4 shape after Inject_n4: {n4.shape}")

        n4 = self.Rep_n4(n4)
        # print(f"n4 shape after Rep_n4: {n4.shape}")

        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        # print(f"n5_adjacent_info shape: {n5_adjacent_info.shape}")

        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        # print(f"n5 shape after Inject_n5: {n5.shape}")

        n5 = self.Rep_n5(n5)
        # print(f"n5 shape after Rep_n5: {n5.shape}")

        outputs = [p3, n4, n5]

        return outputs
#
# from timm.models.mobilevit import MobileVitBlock, MobileVitV2Block
#
# @MODELS.register_module()
# class MobileVitBlock_repvit(RepGDNeck):
#     def __init__(
#             self,
#             channels_list=None,
#             num_repeats=None,
#             block=MobileVitV2Block,
#             extra_cfg=None
#     ):
#         self.block1=block
#         super().__init__(
#             channels_list=channels_list,
#             num_repeats=num_repeats,
#             block=RepVGGBlock,
#             extra_cfg=extra_cfg
#         )
#         print(self.block1)
#         self.low_IFM = nn.Sequential(
#             Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
#             *[self.block1(extra_cfg.embed_dim_p) for _ in range(extra_cfg.fuse_block_num)],
#             Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]),
#                  kernel_size=1, stride=1, padding=0),
#         )
#
#         self.Rep_p4 = nn.Sequential(*(self.block1(channels_list[5]) for _ in range(num_repeats[0])))
#
#         self.Rep_p3 = nn.Sequential(*(self.block1(channels_list[6]) for _ in range(num_repeats[1])))
#
#         self.Rep_n4 = nn.Sequential(*(self.block1(channels_list[6] + channels_list[7]) for _ in range(num_repeats[2])))
#
#         self.Rep_n5 = nn.Sequential(*(self.block1(channels_list[5] + channels_list[9]) for _ in range(num_repeats[3])))
#
# #
# #
# # # 创建 RepGDNeck 实例
# import torch
# import torch.nn as nn
#
# class ExtraConfig:
#     def __init__(self, cfg):
#         for key, value in cfg.items():
#             setattr(self, key, value)
#
# # 创建 RepGDNeck 实例
# extra_cfg_dict = {
#     'norm_cfg': {'type': 'BN', 'requires_grad': True},
#     'depths': 2,
#     'fusion_in': 1248,
#     'fusion_act': {'type': 'ReLU6'},
#     'fuse_block_num':2,
#     'embed_dim_p': 128,
#     'embed_dim_n': 1088,
#     'key_dim': 8,
#     'num_heads': 4,
#     'mlp_ratios': 1,
#     'attn_ratios': 2,
#     'c2t_stride': 2,
#     'drop_path_rate': 0.1,
#     'trans_channels': [80, 48, 96, 176],
#     # 'trans_channels': [64, 32, 64, 128],
#
#     'pool_mode': 'torch'
# }
#
# extra_cfg = ExtraConfig(extra_cfg_dict)
#
# model = RepGDNeck(
#
#     # channels_list=[48, 48, 96, 192, 384, 96, 48, 48, 96, 96, 192],
#     # channels_list=[32,  32,  64,  96,  960, 128, 64, 64, 128, 128, 256],
#     channels_list=[32,  48,  80,  160,  960, 80, 48,48, 96, 96, 176],
#
#     # channels_list=[16, 32, 64, 128, 256, 64, 32, 32, 64, 64, 128],
#     num_repeats=[2, 2,2, 2],
#     block=RepVGGBlock,
#     extra_cfg=extra_cfg
# )
#
# # 创建示例输入
# input = [
#     torch.randn(1, 48, 240, 240),
#     torch.randn(1, 80, 120, 120),
#     torch.randn(1,160, 60, 60),
#     torch.randn(1, 960, 30, 30)
# ]
# # 前向传递测试
# output = model(input)
# f = f'{model._get_name()}_mobile_small.onnx'
# import os
# torch.onnx.export(model, input, f)
# os.system(f'onnxslim {f} {f} && open {f}')
#
# # 打印输出
# for i, out in enumerate(output):
#     print(f'Output {i}: {out.shape}')
