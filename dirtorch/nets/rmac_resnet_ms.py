import pdb
from .rmac_resnet import *


class ResNet_RMAC_MultiScale (ResNet_RMAC):
    """ ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, out_dim=2048, scales=[1,0.5], **kwargs):
        ResNet_RMAC.__init__(self, block, layers, out_dim, **kwargs)
        assert scales[0] == 1 and all(0<s<1 for s in scales[1:])
        self.scales = scales

    def forward(self,x):
        h, w = x.size()[-2:]

        res = ResNet_RMAC.forward(self,x) # at original scale
        for scale in self.scales[1:]:
            interp_input = nn.Upsample(size=(int(0.5+h*scale), int(0.5+w*scale)), mode='bilinear', align_corners=False)
            res += ResNet_RMAC.forward(self,interp_input(x))

        res /= len(self.scales)
        return l2_normalize(res)





def resnet18_rmac_ms(scales, **kwargs):
    return resnet18_rmac(backbone=ResNet_RMAC_MultiScale, scales=scales, **kwargs)

def resnet50_rmac_ms(scales, **kwargs):
    return resnet50_rmac(backbone=ResNet_RMAC_MultiScale, scales=scales, **kwargs)

def resnet101_rmac_ms(scales, **kwargs):
    return resnet101_rmac(backbone=ResNet_RMAC_MultiScale, scales=scales, **kwargs)

def resnet152_rmac_ms(scales, **kwargs):
    return resnet152_rmac(backbone=ResNet_RMAC_MultiScale, scales=scales, **kwargs)









