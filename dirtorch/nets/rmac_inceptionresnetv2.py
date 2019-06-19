import pdb
from .backbones.inceptionresnetv2 import *
from .layers.pooling import GeneralizedMeanPooling, GeneralizedMeanPoolingP


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class InceptionResNetV2_RMAC(InceptionResNetV2):
    """ ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, out_dim=2048, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0,
                       dropout_p=None, without_fc=False):
        InceptionResNetV2.__init__(self, 0)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling.startswith('gem'):
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(1536, out_dim)
        self.fc_name = 'last_linear'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        x = InceptionResNetV2.forward(self, x)

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




def inceptionresnetv2_rmac(backbone=InceptionResNetV2_RMAC, **kwargs):
    kwargs.pop('scales', None)
    return backbone(**kwargs)


