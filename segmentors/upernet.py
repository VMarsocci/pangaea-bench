import torch.nn.functional as F
import torch
import torch.nn as nn


from .ltae import LTAE2d

from utils.registry import SEGMENTOR_REGISTRY

@SEGMENTOR_REGISTRY.register()
class UPerNet(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6), feature_multiplier: int = 1):
        super().__init__()

        # self.frozen_backbone = frozen_backbone

        self.model_name = 'UPerNet'
        self.encoder = encoder
        self.finetune = args.finetune
        self.feature_multiplier = feature_multiplier

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        #
        # if frozen_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

        self.neck = Feature2Pyramid(embed_dim=cfg['in_channels'] * feature_multiplier, rescales=[4, 2, 1, 0.5])

        self.align_corners = False

        self.in_channels = [cfg['in_channels'] * feature_multiplier for _ in range(4)]
        self.channels = cfg['channels']
        self.num_classes = 1 if cfg['binary'] else cfg['num_classes']

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                      out_channels=self.channels,
                      kernel_size=3,
                      padding=1),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True)
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=self.channels,
                          kernel_size=1,
                          padding=0),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False)
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(in_channels=self.channels,
                          out_channels=self.channels,
                          kernel_size=3,
                          padding=1),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False)
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=len(self.in_channels) * self.channels,
                          out_channels=self.channels,
                          kernel_size=3,
                          padding=1),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=True)
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        #inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, img, output_shape=None):
        """Forward function."""
        # if self.freezed_backbone:
        #     with torch.no_grad():
        #         feat = self.backbone(img)
        # else:
        if not self.finetune:
            with torch.no_grad():
                feat = self.encoder(img)
        else:
            feat = self.encoder(img)
        #print(feat)

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        #fixed bug just for optical single modality
        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output

@SEGMENTOR_REGISTRY.register()
class MTUPerNet(UPerNet):
    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6)):
        super().__init__(args, cfg, encoder, pool_scales=(1, 2, 3, 6))   

        self.multi_temporal = cfg["multi_temporal"]
        self.multi_temporal_strategy = cfg["multi_temporal_strategy"]
        if self.encoder.model_name in ["satlas_pretrain"]:
            self.multi_temporal_strategy = None
        if self.multi_temporal_strategy == "ltae":
            self.tmap = LTAE2d(positional_encoding=False, in_channels=cfg['in_channels'],
                                        mlp=[cfg['in_channels'], cfg['in_channels']], d_model=cfg['in_channels'])
        elif self.multi_temporal_strategy == "linear":
            self.tmap = nn.Linear(self.multi_temporal, 1)
        else:
            self.tmap = None

    def forward(self, img, output_shape=None):
        """Forward function for change detection."""

        if self.encoder.model_name in ["Prithvi", "satlas_pretrain"]:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
        else:
            feats = []
            for i in range(self.multi_temporal):
                if not self.finetune:
                    with torch.no_grad():
                        if self.encoder.model_name in ["SpectralGPT"]:
                            feats.append(self.encoder({k: v[:,:,[i],:,:] for k, v in img.items()}))
                        else:
                            feats.append(self.encoder({k: v[:,:,i,:,:] for k, v in img.items()}))
                else:
                    if self.encoder.model_name in ["SpectralGPT"]:
                        feats.append(self.encoder({k: v[:,:,[i],:,:] for k, v in img.items()}))
                    else:
                        feats.append(self.encoder({k: v[:,:,i,:,:] for k, v in img.items()}))  

            feats = [list(i) for i in zip(*feats)]                
            feats = [torch.stack(feat_layers, dim = 2) for feat_layers in feats]
        
        if self.encoder.model_name not in ["satlas_pretrain"]:
            for i in range(len(feats)):
                if self.multi_temporal_strategy == "ltae":
                    feats[i] = self.tmap(feats[i])
                elif self.multi_temporal_strategy == "linear":
                    feats[i] = self.tmap(feats[i].permute(0, 1, 3, 4, 2)).squeeze(-1)
        
        feat = self.neck(feats)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output


class SiamUPerNet(UPerNet):
    def __init__(self, args, cfg, encoder, pool_scales, strategy):

        self.strategy = strategy
        if self.strategy == 'diff':
            self.feature_multiplier = 1
        elif self.strategy == 'concat':
            self.feature_multiplier = 2
        else:
            raise NotImplementedError
        
        super().__init__(args, cfg, encoder, pool_scales=pool_scales, feature_multiplier=self.feature_multiplier) 
        
    def forward(self, img, output_shape=None):
        """Forward function for change detection."""

        if self.encoder.model_name != "Prithvi":
            if self.encoder.model_name == "SpectralGPT":
                # Retains the temporal dimension
                img1 = {k: v[:,:,[0],:,:] for k, v in img.items()}
                img2 = {k: v[:,:,[1],:,:] for k, v in img.items()}
            else:
                img1 = {k: v[:,:,0,:,:] for k, v in img.items()}
                img2 = {k: v[:,:,1,:,:] for k, v in img.items()}

            if not self.finetune:
                with torch.no_grad():
                    feat1 = self.encoder(img1)
                    feat2 = self.encoder(img2)
            else:
                feat1 = self.encoder(img1)
                feat2= self.encoder(img2)

        else:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
        
            feat1, feat2 = [], []
            for i in range(len(feats)):
                feat1.append(feats[i][:,:,0, :, :].squeeze(2))
                feat2.append(feats[i][:,:,1, :, :].squeeze(2))
 
        if self.strategy == 'diff':
            feat = [f2 - f1 for f1, f2 in zip(feat1, feat2)]
        elif self.strategy == 'concat':
            feat = [torch.concat((f1, f2), dim=1) for f1, f2 in zip(feat1, feat2)]
        else:
            raise NotImplementedError

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output


@SEGMENTOR_REGISTRY.register()
class SiamDiffUPerNet(SiamUPerNet):
    # Siamese UPerNet for change detection with feature differencing strategy
    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6)):
        super().__init__(args, cfg, encoder, pool_scales, 'diff')


@SEGMENTOR_REGISTRY.register()
class SiamConcUPerNet(SiamUPerNet):
    # Siamese UPerNet for change detection with feature concatenation strategy
    def __init__(self, args, cfg, encoder, pool_scales=(1, 2, 3, 6)):
        super().__init__(args, cfg, encoder, pool_scales, 'concat')


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.channels,
                              kernel_size=1,
                              padding=0),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True)
                    ))


    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)
