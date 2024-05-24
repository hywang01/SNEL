import torchvision
import torch.nn as nn


def densenet121(pretrained=True, **kwargs):
    if pretrained:
        model = torchvision.models.densenet121(pretrained=True)
    else:
        model = torchvision.models.densenet121(pretrained=False)
        
    #model.classifier = nn.Linear(model.classifier.in_features, 26)

    return model
