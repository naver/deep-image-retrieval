import pdb
from .backbones.senet import *
from .layers.pooling import GeneralizedMeanPooling, GeneralizedMeanPoolingP


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class SENet_RMAC(SENet):
    """ SENet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, groups, reduction, model_name,
                 out_dim=2048, norm_features=False, pooling='gem', gemp=3, center_bias=0,
                 dropout_p=None, without_fc=False, **kwargs):
        SENet.__init__(self, block, layers, groups, reduction, 0, model_name, **kwargs)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling == 'gem':
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, out_dim)
        self.fc_name = 'last_linear'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        x = SENet.forward(self, x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.detach:
            # stop the back-propagation here, if needed
            x = Variable(x.detach())
            x = self.id(x)  # fake transformation

        if self.center_bias > 0:
            b = self.center_bias
            bias = 1 + torch.FloatTensor([[[[0,0,0,0],[0,b,b,0],[0,b,b,0],[0,0,0,0]]]]).to(x.device)
            bias = torch.nn.functional.interpolate(bias, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = x*bias

        # global pooling
        x = self.adpool(x)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.last_linear(x)

        x = l2_normalize(x, axis=-1)
        return x



def senet154_rmac(backbone=SENet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    return backbone(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16, model_name='senet154', **kwargs)

def se_resnet50_rmac(backbone=SENet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    kwargs = {'inplanes': 64, 'input_3x3': False, 'downsample_kernel_size': 1, 'downsample_padding': 0, **kwargs}
    return backbone(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16, model_name='se_resnet50', **kwargs)

def se_resnet101_rmac(backbone=SENet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    kwargs = {'inplanes': 64, 'input_3x3': False, 'downsample_kernel_size': 1, 'downsample_padding': 0, **kwargs}
    return backbone(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16, model_name='se_resnet101', **kwargs)

def se_resnet152_rmac(backbone=SENet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    kwargs = {'inplanes': 64, 'input_3x3': False, 'downsample_kernel_size': 1, 'downsample_padding': 0, **kwargs}
    return backbone(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16, model_name='se_resnet152', **kwargs)

def se_resnext50_32x4d_rmac(backbone=SENet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    kwargs = {'inplanes': 64, 'input_3x3': False, 'downsample_kernel_size': 1, 'downsample_padding': 0, **kwargs}
    return backbone(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, model_name='se_resnext50_32x4d', **kwargs)

def se_resnext101_32x4d_rmac(backbone=SENet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    kwargs = {'inplanes': 64, 'input_3x3': False, 'downsample_kernel_size': 1, 'downsample_padding': 0, **kwargs}
    return backbone(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16, model_name='se_resnext101_32x4d', **kwargs)
