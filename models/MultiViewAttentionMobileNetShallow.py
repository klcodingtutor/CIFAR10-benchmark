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

    def forward(self, x, return_att_map=True):
        # Get the latent features from each model
        pretrained_latents = []
        not_trained_latents = []
        pretrained_att_maps = []
        not_trained_att_maps = []

        for model in self.pretrained_models:
            #  x, att_map, x_att, latent= model(x, return_att_map=True, return_latent=True)
            x, att_map, x_att, latent = model(x, return_att_map=True, return_latent=True)
            pretrained_latents.append(latent)
            pretrained_att_maps.append(att_map)
        
        for model in self.not_trained_models:
            x, att_map, x_att, latent = model(x, return_att_map=True, return_latent=True)
            not_trained_latents.append(latent)
            not_trained_att_maps.append(att_map)
        
        # Concatenate the latent features
        pretrained_latents = torch.cat(pretrained_latents, dim=1)
        not_trained_latents = torch.cat(not_trained_latents, dim=1)
        # Concatenate the attention maps
        pretrained_att_maps = torch.cat(pretrained_att_maps, dim=1)
        not_trained_att_maps = torch.cat(not_trained_att_maps, dim=1)

        # Concatenate the latent features
        latent = torch.cat((pretrained_latents, not_trained_latents), dim=1)
        # Pass the concatenated latent features through the fusion layer
        x = self.fusion_layer(latent)
        if return_att_map:
            return x, pretrained_att_maps, not_trained_att_maps
        else:
            return x

    def freeze_pretrained_models(self):
        # Freeze the pretrained models
        for model in self.pretrained_models:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_pretrained_models(self):
        # Unfreeze the pretrained models
        for model in self.pretrained_models:
            for param in model.parameters():
                param.requires_grad = True

    def freeze_not_trained_models(self):
        # Freeze the not trained models
        for model in self.not_trained_models:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_not_trained_models(self):
        # Unfreeze the not trained models
        for model in self.not_trained_models:
            for param in model.parameters():
                param.requires_grad = True
    
    def freeze_fusion_layer(self):
        # Freeze the fusion layer
        for param in self.fusion_layer.parameters():
            param.requires_grad = False

    def unfreeze_fusion_layer(self):
        # Unfreeze the fusion layer
        for param in self.fusion_layer.parameters():
            param.requires_grad = True

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
    
