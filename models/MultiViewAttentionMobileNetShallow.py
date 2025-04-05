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

        # Get the feature dimension from the first pretrained model
        feature_dim = pretrained_models[0].fc.in_features
        num_features = len(pretrained_models) + len(not_trained_models)  # Total number of feature sources

        # Define the attention-based fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * num_features, feature_dim),  # Project concatenated features
            nn.ReLU(),
            nn.Linear(feature_dim, num_features),  # Compute attention scores
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim, n_classes),
        )

    def forward(self, x, return_att_map=True, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        x_input = x.clone()

        # Initialize lists for latent features and attention maps
        pretrained_latents = []
        not_trained_latents = []
        pretrained_att_maps = []
        not_trained_att_maps = []

        # Extract features from pretrained models
        for i, model in enumerate(self.pretrained_models):
            x = x_input.clone().to(device)
            x, att_map, x_att, latent = model(x, return_att_map=True, return_latent=True)
            pretrained_latents.append(latent)
            pretrained_att_maps.append(att_map)
    
        # Extract features from not-trained models
        for i, model in enumerate(self.not_trained_models):
            x = x_input.clone().to(device)
            x, att_map, x_att, latent = model(x, return_att_map=True, return_latent=True)
            not_trained_latents.append(latent)
            not_trained_att_maps.append(att_map)

        # Combine all latent features
        all_latents = pretrained_latents + not_trained_latents
        if not all_latents:
            raise ValueError("No latent features extracted. Check your models.")

        # Stack all latent features into a tensor: (batch_size, num_features, feature_dim)
        print(f"all_latents: {len(all_latents)}]")
        print(f"all_latents [0]: {all_latents[0].shape}]")
        latents = torch.stack(all_latents, dim=1)  # (batch_size, num_features, feature_dim)
        print(f"latents (After Stack).shape: {latents.shape}]")
        batch_size = latents.size(0)

        # Attention-based fusion
        flat_latents = latents.view(batch_size, -1)  # (batch_size, num_features * feature_dim)
        print(f"flat_latents.shape: {flat_latents.shape}]")
        attn_scores = self.fusion_layer(flat_latents)  # (batch_size, num_features)
        print(f"attn_scores.shape: {attn_scores.shape}]")
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch_size, num_features, 1)
        print(f"attn_weights.shape: {attn_weights.shape}]")
        print(f"attn_weights: {attn_weights}]")
        fused_latent = (latents * attn_weights).sum(dim=1)  # (batch_size, feature_dim)
        print(f"fused_latent.shape: {fused_latent.shape}]")

        # Final classification
        x = self.classifier(fused_latent)

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
        # Freeze the classifier layer
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_fusion_layer(self):
        # Unfreeze the fusion layer
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
        # Unfreeze the classifier layer
        for param in self.classifier.parameters():
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
    
