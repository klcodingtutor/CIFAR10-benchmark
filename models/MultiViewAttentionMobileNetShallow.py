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

        # Lightweight attention: Squeeze-and-Excitation style
        reduction = 16  # Reduction factor to reduce parameters
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),  # Squeeze
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, num_features),  # Excitation (one weight per feature)
            nn.Sigmoid()  # Output weights between 0 and 1
        )
        # Initialize the attention weights with Kaiming uniform distribution
        nn.init.kaiming_uniform_(self.attention[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.attention[2].weight, nonlinearity='sigmoid')

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim, n_classes),
        )
        
        # Initialize the classifier weights
        nn.init.kaiming_uniform_(self.classifier[1].weight, nonlinearity='relu')

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

        # Stack latents: (batch_size, num_features, feature_dim)
        latents = torch.stack(all_latents, dim=1)  # (batch_size, num_features, feature_dim)
        batch_size = latents.size(0)

        # Compute a global descriptor for attention
        global_descriptor = latents.mean(dim=1)  # (batch_size, feature_dim) - average across features
        attn_weights = self.attention(global_descriptor)  # (batch_size, num_features)
        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, num_features, 1)

        # Apply attention weights and fuse
        fused_latent = (latents * attn_weights).sum(dim=1)  # (batch_size, feature_dim)

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
        print(f"Freezed {len(self.pretrained_models)} pretrained models.")
    
    def unfreeze_pretrained_models(self):
        # Unfreeze the pretrained models
        for model in self.pretrained_models:
            for param in model.parameters():
                param.requires_grad = True
        print(f"Unfreezed {len(self.pretrained_models)} pretrained models.")

    def freeze_not_trained_models(self):
        # Freeze the not trained models
        for model in self.not_trained_models:
            for param in model.parameters():
                param.requires_grad = False
        print(f"Freezed {len(self.not_trained_models)} not trained models.")
    
    def unfreeze_not_trained_models(self):
        # Unfreeze the not trained models
        for model in self.not_trained_models:
            for param in model.parameters():
                param.requires_grad = True
        print(f"Unfreezed {len(self.not_trained_models)} not trained models.")
    
    def freeze_fusion_layer(self):
        # Freeze the fusion layer
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("Fusion layer is frozen. [Attention and Classifier]")

    def unfreeze_fusion_layer(self):
        # Unfreeze the fusion layer
        for param in self.attention.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Fusion layer is unfrozen. [Attention and Classifier]")
        

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
    
