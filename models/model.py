from torch import nn
from transformers import AutoModelForImageClassification, \
    AutoFeatureExtractor, ResNetForImageClassification, EfficientNetForImageClassification, AutoImageProcessor
from vit_model import *


from .cait import CaiTWrapper


def get_model_with_preprocessor(model_name, device="cpu"):
    if "cait" in model_name:
        # cait uses vit defaults
        preprocessor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        model = CaiTWrapper()
    else:
        preprocessor = AutoFeatureExtractor.from_pretrained(model_name)
        #preprocessor = AutoImageProcessor.from_pretrained('google/efficientnet-b0')
        #model = EfficientNetForImageClassification.from_pretrained('google/efficientnet-b0')
        #model = AutoModelForImageClassification.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=1)
        #preprocessor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-50')
        #model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
        in_features = model.classifier.in_features
        # replace classifier with linear
        model.classifier = nn.Linear(in_features, 1)
        #model.classifier[1] = nn.Linear(2048, 1)
    # set backbone
    if hasattr(model, "vit"):
        print()
        #model.backbone = model.vit
    elif hasattr(model, "beit"):
        model.backbone = model.beit
    model = model.to(device)
    return model, preprocessor
