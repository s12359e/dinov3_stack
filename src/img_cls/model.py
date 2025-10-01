"""
Building a linear classifier on top of DINOv3 backbone.
"""

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

def load_model(weights: str=None, model_name: str=None, repo_dir: str=None):
    """Load a backbone using Hugging Face transformers AutoModel."""

    if weights is not None:
        print('Loading pretrained backbone weights from Hugging Face: ', weights)
        model = AutoModel.from_pretrained(weights, trust_remote_code=True)
    else:
        if model_name is None:
            raise ValueError('Either `weights` or `model_name` must be provided.')
        print('No pretrained weights path given. Initializing from config: ', model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_config(config)

    return model

# def build_model(
#     num_classes: int=10, 
#     fine_tune: bool=False, 
#     weights: str=None, 
#     model_name: str=None,
#     repo_dir: str=None
# ):
#     backbone_model = load_model(
#         weights=weights, model_name=model_name, repo_dir=repo_dir
#     )

#     model = torch.nn.Sequential(OrderedDict([
#         ('backbone', backbone_model),
#         ('head', torch.nn.Linear(
#             in_features=backbone_model.norm.normalized_shape[0], 
#             out_features=num_classes, 
#             bias=True
#         ))
#     ]))
    
#     if not fine_tune:
#         for params in model.backbone.parameters():
#             params.requires_grad = False

#     return model

class Dinov3Classification(nn.Module):
    def __init__(
        self, 
        fine_tune: bool=False, 
        num_classes: int=2,
        weights: str=None,
        model_name: str=None,
        repo_dir: str=None
    ):
        super(Dinov3Classification, self).__init__()

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )

        self.head = nn.Linear(
            in_features=self.backbone_model.norm.normalized_shape[0], 
            out_features=num_classes, 
            bias=True
        )

        if not fine_tune:
            for params in self.backbone_model.parameters():
                params.requires_grad = False

    def forward(self, x):
        features = self.backbone_model(x)
        # Through classifier head.
        classifier_out = self.head(features)
        return classifier_out

if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    from torchinfo import summary

    import numpy as np

    sample_size = 224

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(
            sample_size, 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(sample_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
    ])

    # Loading the pretrained model without classification head.
    model = load_model(weights='facebook/dinov2-base')

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Testing forward pass.
    pil_image = Image.fromarray(np.ones((sample_size, sample_size, 3), dtype=np.uint8))
    model_input = transform(pil_image).unsqueeze(0)

    # summary(
    #     model,
    #     input_data=model_input,
    #     col_names=('input_size', 'output_size', 'num_params'),
    #     row_settings=['var_names']
    # )

    # Manual torch forward pass.
    with torch.no_grad():
        features = model.forward_features(model_input)
        patch_features = features['x_norm_patchtokens']

    print(features.keys())
    print(f"Patch features shape: {patch_features.shape}")

    # Check the forward passes through the complete model.
    # To check what gets fed to the classification layer.
    print(model)

    model_cls = Dinov3Classification(
        weights='facebook/dinov2-base',
        model_name='facebook/dinov2-base'
    )

    summary(
        model_cls,
        input_data=model_input,
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )

    features = model_cls.backbone_model(model_input)
    print(f"Shape of features getting fed to classification layer: {features.shape}")