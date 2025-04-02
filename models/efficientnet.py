from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def get_efficientnet(model_name='efficientnet-b0', num_classes=10, pretrained=False, transfer_learning=False):
    if pretrained:
        model = EfficientNet.from_pretrained(model_name)
    else:
        model = EfficientNet.from_name(model_name)
    
    # Modify the classifier for CIFAR-10
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    
    # Freeze pre-trained layers for transfer learning
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
        # Ensure the classifier remains trainable
        for param in model._fc.parameters():
            param.requires_grad = True
    
    return model