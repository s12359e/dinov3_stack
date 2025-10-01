import torch
import torch.nn as nn

from torchvision.models.detection.ssd import (
    SSD,
    DefaultBoxGenerator,
    SSDHead
)
from torchvision.models.detection.retinanet import (
    RetinaNet, RetinaNetHead, AnchorGenerator
)
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

class Dinov3Backbone(nn.Module):
    def __init__(self, 
        weights: str=None,
        model_name: str=None,
        repo_dir: str=None,
        fine_tune: bool=False
    ):
        super(Dinov3Backbone, self).__init__()

        self.model_name = model_name

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )

        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.backbone_model.get_intermediate_layers(
            x, 
            n=1, 
            reshape=True, 
            return_class_token=False, 
            norm=True
        )[0]

        return out

# class Dinov3Detection(nn.Module):
#     def __init__(
#         self, 
#         fine_tune: bool=False, 
#         num_classes: int=2,
#         weights: str=None,
#         model_name: str=None,
#         repo_dir: str=None,
#         resolution: list=[640, 640],
#         nms: float=0.45,
#         feature_extractor: str='last' # OR 'multi'
#     ):

#         super(Dinov3Detection, self).__init__()

#         self.backbone = Dinov3Backbone(
#             weights=weights, 
#             model_name=model_name, 
#             repo_dir=repo_dir, 
#             fine_tune=fine_tune
#         )
       
#         self.num_classes = num_classes

#         out_channels = [768, 768, 768, 768, 768, 768]
#         anchor_generator = DefaultBoxGenerator(
#             [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#         )
#         num_anchors = anchor_generator.num_anchors_per_location()

#         head = SSDHead(out_channels, num_anchors, num_classes=num_classes)

#         self.model = SSD(
#             backbone=self.backbone,
#             num_classes=num_classes,
#             anchor_generator=anchor_generator,
#             size=resolution,
#             head=head,
#             nms_thresh=nms
#         )
    
#     def forward(self, x):
#         out = self.model(x)

#         return out

def dinov3_detection(
    fine_tune: bool=False, 
    num_classes: int=2,
    weights: str=None,
    model_name: str=None,
    repo_dir: str=None,
    resolution: list=[640, 640],
    nms: float=0.45,
    feature_extractor: str='last', # OR 'multi'
    head: str='ssd' # Detection head type, ssd or retinanet
):
    backbone = Dinov3Backbone(
        weights=weights, 
        model_name=model_name, 
        repo_dir=repo_dir, 
        fine_tune=fine_tune
    )

    if head == 'ssd':
        out_channels = [backbone.backbone_model.norm.normalized_shape[0]] * 6
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )
        ###########################################
        # out_channels = [backbone.backbone_model.norm.normalized_shape[0]] * 6
        # anchor_generator = DefaultBoxGenerator(
            # aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            # steps=[16, 32, 64, 100, 300, 600]
        # )
        
        num_anchors = anchor_generator.num_anchors_per_location()
        det_head = SSDHead(out_channels, num_anchors, num_classes=num_classes)
    
        model = SSD(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            size=resolution,
            head=det_head,
            nms_thresh=nms
        )
    
    elif head == 'retinanet':
        backbone.out_channels = backbone.backbone_model.norm.normalized_shape[0]
        anchor_sizes = ((32, 64, 128, 256, 512),)  # one tuple, for one feature map
        aspect_ratios = ((0.5, 1.0, 2.0),)         # one tuple, same idea
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        model = RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            min_size=resolution[0],
            max_size=resolution[1],
            # head=head,
            nms_thresh=nms
        )

    return model


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    from torchinfo import summary

    import numpy as np

    input_size = 640

    transform = transforms.Compose([
        transforms.Resize(
            input_size, 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
    ])

    model_names = [
        'facebook/dinov2-small',
        'facebook/dinov2-base',
    ]

    for head in ['ssd', 'retinanet']:
        print(f"Building {head} models...\n\n")
        for model_name in model_names:
            print('Testing: ', model_name)
            model = dinov3_detection(
                weights=model_name,
                model_name=model_name,
                feature_extractor='last', # OR 'last'
                head=head
            )
            model.eval()
            print(model)
        
            random_image = Image.fromarray(np.ones(
                (input_size, input_size, 3), dtype=np.uint8)
            )
            x = transform(random_image).unsqueeze(0)
        
            with torch.no_grad():
                outputs = model(x)
            
            print(outputs)
        
            summary(
                model, 
                input_data=x,
                col_names=('input_size', 'output_size', 'num_params'),
                row_settings=['var_names'],
            )
            print('#' * 50, '\n\n')