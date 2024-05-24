import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

from .resnet import resnet18, resnet50
from .densenet import densenet121


def build_model_training(cfg):
    if cfg.models.baseline_only:
        model = BaselineModel(cfg)
    elif cfg.models.model_name == 'snel':
        model = SNELModel(cfg)
    else:
        model = BaselineModel(cfg)

    return model


class BaselineModel(BaseModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # set up backbone
        if cfg.models.backbone == 'resnet18':
            self.backbone = resnet18()
        elif cfg.models.backbone == 'resnet50':
            self.backbone = resnet50()
        elif cfg.models.backbone == 'densenet121':
            self.backbone = densenet121()
        else:
            print('A model name is required')

        if cfg.models.backbone == 'densenet121':
            fdim = self.backbone.classifier.in_features
            self._fdim = fdim
        else:
            fdim = self.backbone.out_features
            self._fdim = fdim

        # set up classifier
        num_classes = cfg.datasets.num_classes     
        self.classifier = None
        
        if num_classes > 0:
            if cfg.models.backbone == 'densenet121':
                self.backbone.classifier = nn.Linear(fdim, num_classes)
            else:
                self.classifier = nn.Linear(fdim, num_classes)

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x):
        if self.cfg.models.backbone == 'densenet121':
            y_logit = self.backbone(x) 
            f = self.backbone.features(x)
            out = F.relu(f, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            self._feature = out
        else:
            f = self.backbone(x)
            self._feature = f

            y_logit = self.classifier(f)

        return y_logit

    def return_feature(self):

        return self._feature


class SNELModel(BaseModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # set up backbone
        if cfg.models.backbone == 'resnet18':
            self.backbone = resnet18()
        elif cfg.models.backbone == 'resnet50':
            self.backbone = resnet50()
        elif cfg.models.backbone == 'densenet121':
            self.backbone = densenet121()
        else:
            print('A model name is required')

        if cfg.models.backbone == 'densenet121':
            fdim = self.backbone.classifier.in_features
            self._fdim = fdim
        else:
            fdim = self.backbone.out_features
            self._fdim = fdim

        # set up classifier
        num_classes = cfg.datasets.num_classes
        self.classifier = None
        
        if num_classes > 0:
            if cfg.models.backbone == 'densenet121':
                self.backbone.classifier = nn.Linear(fdim, num_classes)
                self.classifier = MCdropClassifier(
                    in_features = fdim,
                    num_classes = num_classes,
                    bottleneck_dim = cfg.models.bottleneck_dim,
                    dropout_rate = cfg.models.dropout_rate,
                    dropout_type = cfg.models.dropout_type
                )
            else:
                self.classifier = MCdropClassifier(
                                    in_features = fdim,
                                    num_classes = num_classes,
                                    bottleneck_dim = cfg.models.bottleneck_dim,
                                    dropout_rate = cfg.models.dropout_rate,
                                    dropout_type = cfg.models.dropout_type
                                )
                
    @property
    def fdim(self):
        return self._fdim

    def forward(self, x):
        if self.cfg.models.backbone == 'densenet121':
            f = self.backbone.features(x)
            out = F.relu(f, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            self._feature = out
            
            y_logit = self.classifier(out)
        else:
            f = self.backbone(x)
            self._feature = f

            y_logit = self.classifier(f)

        return y_logit

    def return_feature(self):

        return self._feature


class MCdropClassifier(nn.Module):
    def __init__(self, in_features, num_classes, 
                 bottleneck_dim=512, dropout_rate=0.5, 
                 dropout_type='Bernoulli'):
        super(MCdropClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        self.bottleneck_drop = self._make_dropout(dropout_rate, dropout_type)

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.ReLU(),
            self.bottleneck_drop
        )

        self.prediction_layer = nn.Linear(bottleneck_dim, num_classes)

    def _make_dropout(self, dropout_rate, dropout_type):
        if dropout_type == 'Bernoulli':
            return nn.Dropout(dropout_rate)
        elif dropout_type == 'Gaussian':
            return GaussianDropout(dropout_rate)
        else:
            raise ValueError(f'Dropout type not found')

    def activate_dropout(self):
        self.bottleneck_drop.train()

    def forward(self, x):
        hidden = self.bottleneck_layer(x)
        pred = self.prediction_layer(hidden)
        return pred


class GaussianDropout(nn.Module):
    def __init__(self, drop_rate):
        super(GaussianDropout, self).__init__()
        self.drop_rate = drop_rate
        self.mean = 1.0
        self.std = math.sqrt(drop_rate / (1.0 - drop_rate))

    def forward(self, x):
        if self.training:
            gaussian_noise = torch.randn_like(x, requires_grad=False).to(x.device) * self.std + self.mean
            return x * gaussian_noise
        else:
            return x
            