import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewAttentionMobileNetShallow(nn.Module):
    '''
    MultiViewAttentionMobileNetShallow is a neural network module designed to combine the outputs of multiple 
    AttentionMobileNetShallow models, both pretrained and not trained, into a single fused output. The module 
    extracts latent features from each model, concatenates them, and passes them through a fusion layer to 
    produce the final output.
    Args:
        pretrained_models (list): A list of pretrained AttentionMobileNetShallow models. These models are 
                                  frozen during training.
        not_trained_models (list): A list of not trained AttentionMobileNetShallow models. These models are 
                                   trainable during training.
        n_classes (int): The number of output classes for the final classification.
    Attributes:
        pretrained_models (list): Stores the pretrained models.
        not_trained_models (list): Stores the not trained models.
        n_classes (int): Number of output classes.
        fusion_layer (nn.Linear): A fully connected layer that combines the concatenated latent features 
                                  from all models to produce the final output.
    Methods:
        forward(x, return_att_map=True):
            Performs a forward pass through the module. Extracts latent features and attention maps from 
            each model, concatenates them, and passes the result through the fusion layer.
            Args:
                x (torch.Tensor): Input tensor.
                return_att_map (bool): If True, returns attention maps along with the output.
            Returns:
                torch.Tensor: Final output after the fusion layer.
                torch.Tensor (optional): Attention maps from pretrained models.
                torch.Tensor (optional): Attention maps from not trained models.
        freeze_pretrained_models():
            Freezes the parameters of all pretrained models.
        unfreeze_pretrained_models():
            Unfreezes the parameters of all pretrained models.
        freeze_not_trained_models():
            Freezes the parameters of all not trained models.
        unfreeze_not_trained_models():
            Unfreezes the parameters of all not trained models.
        freeze_fusion_layer():
            Freezes the parameters of the fusion layer.
        unfreeze_fusion_layer():
            Unfreezes the parameters of the fusion layer.
        freeze_all():
            Freezes the parameters of all models and the fusion layer.
        unfreeze_all():
            Unfreezes the parameters of all models and the fusion layer.
    '''

    def __init__(self, pretrained_models, not_trained_models, n_classes):
        super(MultiViewAttentionMobileNetShallow, self).__init__()
        self.pretrained_models = pretrained_models
        self.not_trained_models = not_trained_models
        self.n_classes = n_classes

        # Fusion layer, get from each linear layer of the models
        pretrained_output_size = sum(model.fc.in_features for model in pretrained_models)
        not_trained_output_size = sum(model.fc.in_features for model in not_trained_models)
        self.fusion_layer = nn.Linear(pretrained_output_size + not_trained_output_size, n_classes)
        import math
        nn.init.kaiming_uniform_(self.fusion_layer.weight, a=math.sqrt(5))

    def forward(self, x, return_att_map=True, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Move the input tensor to the specified device
        x = x.to(device)
        # Clone the input tensor to avoid modifying it during the forward pass
        x_input = x.clone()
        # Initialize tensors for latent features and attention maps
        pretrained_latents = None
        not_trained_latents = None
        pretrained_att_maps = None
        not_trained_att_maps = None

        for i, model in enumerate(self.pretrained_models):
            x = x_input.clone().to(device)
            x, att_map, x_att, latent = model(x, return_att_map=True, return_latent=True)
            if i == 0:
                pretrained_latents = latent
                pretrained_att_maps = att_map
            else:
                pretrained_latents = torch.cat((pretrained_latents, latent), dim=1)
                pretrained_att_maps = torch.cat((pretrained_att_maps, att_map), dim=1)
        
        for i, model in enumerate(self.not_trained_models):
            x = x_input.clone().to(device)
            x, att_map, x_att, latent = model(x, return_att_map=True, return_latent=True)
            if i == 0:
                not_trained_latents = latent
                not_trained_att_maps = att_map
            else:
                not_trained_latents = torch.cat((not_trained_latents, latent), dim=1)
                not_trained_att_maps = torch.cat((not_trained_att_maps, att_map), dim=1)
        
        # Concatenate the latent features
        # both exsit
        if pretrained_latents is not None and not_trained_latents is not None:
            latent = torch.cat((pretrained_latents, not_trained_latents), dim=1).to(device)
        # only pretrained exist
        elif pretrained_latents is not None:
            latent = pretrained_latents.to(device)
        # only not trained exist
        elif not_trained_latents is not None:
            latent = not_trained_latents.to(device)
        # neither exist
        else:
            raise ValueError("Both pretrained_latents and not_trained_latents are None. Check your models.")
        
        # Pass the concatenated latent features through the fusion layer
        x = self.fusion_layer(latent)
        if return_att_map:
            return x, pretrained_att_maps.to(device), not_trained_att_maps.to(device)
        else:
            return x

    def freeze_pretrained_models(self):
        # Freeze the pretrained models
        for model in self.pretrained_models:
            for param in model.parameters():
                param.requires_grad = False
        print(f"Freezed {len(self.pretrained_models)} pretrained models")
    
    def unfreeze_pretrained_models(self):
        # Unfreeze the pretrained models
        for model in self.pretrained_models:
            for param in model.parameters():
                param.requires_grad = True
        print(f"Unfreezed {len(self.pretrained_models)} pretrained models")

    def freeze_not_trained_models(self):
        # Freeze the not trained models
        for model in self.not_trained_models:
            for param in model.parameters():
                param.requires_grad = False
        print(f"Freezed {len(self.not_trained_models)} not trained models")
    
    def unfreeze_not_trained_models(self):
        # Unfreeze the not trained models
        for model in self.not_trained_models:
            for param in model.parameters():
                param.requires_grad = True
        print(f"Unfreezed {len(self.not_trained_models)} not trained models")
    
    def freeze_fusion_layer(self):
        # Freeze the fusion layer
        for param in self.fusion_layer.parameters():
            param.requires_grad = False
        print("Freezed fusion layer")

    def unfreeze_fusion_layer(self):
        # Unfreeze the fusion layer
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
        print("Unfreezed fusion layer")

    def freeze_all(self):
        # Freeze all models and fusion layer
        self.freeze_pretrained_models()
        self.freeze_not_trained_models()
        self.freeze_fusion_layer()
    
    def unfreeze_all(self):
        # Unfreeze all models and fusion layer
        self.unfreeze_pretrained_models()
        self.unfreeze_not_trained_models()
        self.unfreeze_fusion_layer()
    
