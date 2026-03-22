from .registry import MODELS
from .hrnet48v2 import HighResolutionNet as HRNet48v2
from .hrnet32v2 import HighResolutionNet as HRNet32v2
from .hrnet18v2 import HighResolutionNet as HRNet18v2
from .multi_task_head import MultitaskHead
from .resnetunet101 import UNetResNetBackbone as ResNetUNet
import os

@MODELS.register("HRNet48v2")
def build_hrnet48(cfg):
    head_size = list(cfg.MODEL.HEAD_SIZE)  # [[2]]
    head_size = [list(sub_list) for sub_list in head_size]
    num_class = sum(sum(head_size, []))  # 2

    model = HRNet48v2(cfg,
                      head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                      num_class=num_class)
    # pretrained = 'hisup/backbones/hrnet_imagenet/hrnetv2_w48_imagenet_pretrained.pth'
    pretrained = 'ldm/models/backbones/hrnet_imagenet/HRNet_W48_C_ssld_pretrained.pth'
    assert os.path.isfile(pretrained), "Pretrained backbone does not exit!"
    model.init_weights(pretrained=pretrained)
    print('INFO:build hrnet-w48-v2 backbone')
    return model

@MODELS.register("HRNet32v2")
def build_hrnet32(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))

    model = HRNet32v2(cfg,
                      head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                      num_class = num_class)

    pretrained = 'hisup/backbones/hrnet_imagenet/hrnetv2_w32_imagenet_pretrained.pth'
    model.init_weights(pretrained=pretrained)
    print('INFO:build hrnet-w32-v2 backbone')
    return model

@MODELS.register("HRNet18v2")
def build_hrnet18(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))

    model = HRNet18v2(cfg,
                      head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                      num_class=num_class)

    pretrained = 'hisup/backbones/hrnet_imagenet/hrnetv2_w18_imagenet_pretrained.pth'
    model.init_weights(pretrained=pretrained)
    print('INFO:build hrnet-w18-v2 backbone')

    return model

@MODELS.register("ResNetUNet101")
def build_resunet101(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))

    model = ResNetUNet(encoder_depth=101, 
                       pretrained=True,
                       head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                       num_class = num_class)

    print('INFO:build ResnetUnet101 backbone')
    return model

def build_backbone(cfg):
    # MODELS:  {'HRNet48v2': <function build_hrnet48 at 0x7f203a876f70>,
    #           'HRNet32v2': <function build_hrnet32 at 0x7f203a88d160>,
    #           'HRNet18v2': <function build_hrnet18 at 0x7f203a88d1f0>,
    #           'ResNetUNet101': <function build_resunet101 at 0x7f203a88d280>}
    assert cfg.MODEL.NAME in MODELS,  \
        "cfg.MODELS.NAME: {} is not registered in registry".format(cfg.MODELS.NAME)

    return MODELS[cfg.MODEL.NAME](cfg)
