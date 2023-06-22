import os
import torch

from torch import nn
from modelUtils import ResNet, Bottleneck, _SimpleSegmentationModel, ASPP, IntermediateLayerGetter, FCNHead


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


def deeplabv3(num_classes=21):
    backbone = ResNet(Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True])
    out_layer = 'layer4'
    out_inplanes = 2048
    aux_layer = 'layer3'
    aux_inplanes = 1024
    return_layers = {out_layer: 'out'}
    return_layers[aux_layer] = 'aux'

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes)
    base_model = DeepLabV3

    model = base_model(backbone, classifier, aux_classifier)

    return model


def createModel(pretrained=True, preChannel=3, channel=3, preNumClasses=21, numClasses=None, weightPath=None, device=None):

    model = deeplabv3(num_classes=preNumClasses)

    if pretrained:
        model.backbone.conv1 = nn.Conv2d(in_channels=preChannel, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=preNumClasses, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=preNumClasses, kernel_size=(1, 1), stride=(1, 1))

        model.load_state_dict(torch.load(os.path.join(weightPath, "weight.pth"), map_location=device))

        for param in model.parameters():
            param.requires_grad = False

        model.backbone.conv1 = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=numClasses, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=numClasses, kernel_size=(1, 1), stride=(1, 1))

    else:
        model.backbone.conv1 = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=numClasses, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=numClasses, kernel_size=(1, 1), stride=(1, 1))
  
    return model


# if __name__ == "__main__":
#     # model = createModel(
#     #     pretrained=True,
#     #     channel=3,
#     #     preNumClasses=21,
#     #     numClasses=21,
#     #     weightPath="/data/sungmin/deeplabv3/originWeight",
#     #     device="cpu"
#     # weightPath='/home/gpuadmin/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'
#     # device=None
#     # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
#     # model.load_state_dict(torch.load(weightPath, map_location=device),strict=True)
#     model = torch.hub.load('yolov5s.pt',map_location='cpu')
#     print(model)
# if __name__ == '__main__':
    
#     # pretrained=True
#     # preChannel=3
#     # channel=3
#     # preNumClasses = 2
#     # numClasses=255
#     # weightPath='/data/sungmin/yolov5/model/yolov5s.pt'
#     # device=None
#     device =map_location='cpu'
#     # model = createModel()
#     # model = torch.load(weightPath, map_location=device)
#     model = torch.load('/data/sungmin/resnet/weight/weight.pth', map_location=device)
#     print(model)
    
    
    
