import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.adaptive_grad_reverse_layer import AdaptiveGradReverse


class FeatureExtractor(nn.Module):
    """Feature extoractor block"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.feature_ex = nn.Sequential(*list(vgg16.children())[:-1])

    def forward(self, inputs):
        inputs = inputs.squeeze(0)
        features = self.feature_ex(inputs)
        features = features.view(features.size(0), -1)
        return features


class ClassPredictor(nn.Module):
    """Class predictor block to Classify the class of Bag using attention mechanism"""
    def __init__(self):
        super(ClassPredictor, self).__init__()
        # full connect layer
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=25088, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        # attention mechanism
        self.attention_mechanism = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # class classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),
        )

    def forward(self, inputs):
        inputs = inputs.squeeze(0)
        H = self.fc_layer(inputs)
        attention = self.attention(H)
        attention = torch.transpose(attention, 1, 0)
        attention = F.softmax(attention, dim=1)
        z = torch.mm(attention, H)  # KxL
        class_y = self.classifier(z)
        return class_y, attention


class DomainPredictor(nn.Module):
    """Domain predictor block to classify domain of patch images"""
    def __init__(self, domain_num):
        super(DomainPredictor, self).__init__()
        # domein classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, domain_num)
        )

    def forward(self, inputs):
        x = inputs.squeeze(0)
        domain_y = self.domain_classifier(x)
        return domain_y


class DAMIL(nn.Module):
    """DA-MIL network"""
    def __init__(self, feature_extractor: FeatureExtractor, class_predictor: ClassPredictor, domain_predictor: DomainPredictor):
        super(DAMIL, self).__init__()
        self.feature_extractor = feature_extractor
        self.class_predictor = class_predictor
        self.domain_predictor = domain_predictor

    def forward(self, input, da_rate):
        x = input.squeeze(0)
        # Extoract feature
        features = self.feature_extractor(x)
        # predict class of bag
        class_y, attention = self.class_predictor(features)
        # predict domains of patch images
        adapGR_features = AdaptiveGradReverse.apply(features, da_rate, attention)  # Input AdaptiveGradReverse
        domain_y = self.domain_predictor(adapGR_features)
        return class_y, domain_y, attention


class MSDAMIL(nn.Module):
    """MS-DA-MIL network"""
    def __init__(self, feature_extractor_scale1: FeatureExtractor, feature_extractor_scale2: FeatureExtractor, class_predictor: ClassPredictor):
        super(MSDAMIL, self).__init__()
        self.feature_extractor_scale1 = feature_extractor_scale1
        self.feature_extractor_scale2 = feature_extractor_scale2
        self.class_predictor = class_predictor
        # update is not required of feature extoractor
        for param in self.feature_extractor_scale1.parameters():
            param.requires_grad = False
        for param in self.feature_extractor_scale2.parameters():
            param.requires_grad = False

    def forward(self, input_scale1, input_scale2):
        x_scale1 = input_scale1.squeeze(0)
        x_scale2 = input_scale2.squeeze(0)
        # Extoract feature from each magnifications 
        features_1 = self.feature_extractor_scale1(x_scale1)
        features_2 = self.feature_extractor_scale1(x_scale2)
        # Concat
        ms_bag = torch.cat([features_1, features_2], dim=0)
        # Predict class of bag
        class_y, attention = self.class_predictor(ms_bag)
        return class_y, attention
