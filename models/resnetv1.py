import torch.nn as nn
import math
import torch

__all__ = ['resnetv1','resnetv1_18', 'resnet_CAMELYON', 'ResNet_512x512_projection_prototype', 'resnet_224x224']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1, num_classes=[1000]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.headcount = len(num_classes)
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            nn.AvgPool2d(7, stride=1),
        ])
        if len(num_classes) == 1:
            self.top_layer = nn.Sequential(nn.Linear(512*4, num_classes[0]))
        else:
            for a, i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(512*4, i))
            self.top_layer = None

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.headcount == 1:
            if self.top_layer:
                out = self.top_layer(out)
            return out
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(out))
            return outp


# for processing 512x512 images: (1) replacing AvgPool2d layer; (2) change fc layer size to 512
class ResNet_512x512(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1, num_classes=[1000]):
        self.inplanes = 64
        super(ResNet_512x512, self).__init__()
        self.headcount = len(num_classes)
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            # nn.AvgPool2d(7, stride=1),
                            nn.AvgPool2d(16, stride=1),
        ])
        if len(num_classes) == 1:
            self.top_layer = nn.Sequential(nn.Linear(512*1, num_classes[0]))
        else:
            for a, i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(512*1, i))
            self.top_layer = None

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False, return_feat_out=False):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.top_layer(feat)
        if return_feat_out:
            return feat, out
        if return_feat:
            return feat
        return out


# for processing 512x512 images: (1) replacing AvgPool2d layer; (2) change fc layer size to 512
class ResNet_512x512_projection_prototype(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], in_channel=3, width=1,
                 output_dim=512, hidden_mlp=2048, nmb_prototypes=300, init=True, normalize=True,
                 eval_mode=False, norm_layer=None
                 ):
        self.inplanes = 64
        super(ResNet_512x512_projection_prototype, self).__init__()
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            # nn.AvgPool2d(7, stride=1),
                            # nn.AvgPool2d(16, stride=1),
                            nn.AdaptiveAvgPool2d(output_size=1),
        ])

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            # self.projection_head = nn.Linear(128*2*2, output_dim)
            self.projection_head = nn.Linear(512, output_dim)
        else:
            self.projection_head = nn.Sequential(
                # nn.Linear(128*2*2, hidden_mlp),
                nn.Linear(512, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            # self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
            print("Multiple Prototypes is not supported now")
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 128 * 2 * 2)
        x = x.view(x.size(0), -1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# for processing 224x224 images: (1) change fc layer size to 512
class ResNet_224x224(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1, num_classes=[1000]):
        self.inplanes = 64
        super(ResNet_224x224, self).__init__()
        self.headcount = len(num_classes)
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            nn.AvgPool2d(7, stride=1),
        ])
        if len(num_classes) == 1:
            self.top_layer = nn.Sequential(nn.Linear(512*1, num_classes[0]))
        else:
            for a, i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(512*1, i))
            self.top_layer = None

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False, return_feat_out=False):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        if self.top_layer is not None:
            out = self.top_layer(feat)
        else:
            out = []
            for i in range(self.headcount):
                out.append(getattr(self, "top_layer%d" % i)(feat))

        if return_feat_out:
            return feat, out
        if return_feat:
            return feat
        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnetv1(num_classes=[1000]):
    """Encoder for instance discrimination and MoCo"""
    return resnet50(num_classes=num_classes)


def resnetv1_18(num_classes=[1000]):
    """Encoder for instance discrimination and MoCo"""
    return resnet18(num_classes=num_classes)


def resnet_CAMELYON():
    return ResNet_512x512(BasicBlock, [2, 2, 2, 2], num_classes=[2])


def resnet_224x224(num_classes=[2]):
    return ResNet_224x224(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    import torch
    # model = resnetv1(num_classes=[500]*3)
    # print([ k.shape for k in model(torch.randn(64,3,224,224))])
    model = ResNet_512x512_projection_prototype()
    print("END")