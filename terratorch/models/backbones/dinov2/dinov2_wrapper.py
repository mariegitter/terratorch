from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import torch
from torch import nn
import numpy as np
import pdb
from torch import Tensor


class DinoV2Wrapper(nn.Module):
    def __init__(self, model: str, ckpt_path: str = None, return_cls_token: bool = True, **kwargs):
        """
        TerraTorch wrapper for DINO V2 models.

        Args:
            model: model name from the ones supported in
                   https://github.com/facebookresearch/dinov2/blob/main/hubconf.py
                   (e.g., "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")
            ckpt_path: local path to the desired DINO V2 model checkpoint downloaded from Meta
            return_cls_token: whether the model should return the class token or not.
                              If True, the class token will be appended to the embedding sequence.

        Returns:
            DinoV2Wrapper model
        """
        super().__init__()

        if ckpt_path is None:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model, pretrained=False)
        else:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model, weights=ckpt_path)

        if hasattr(self.dinov2, "blocks"):
            self.out_channels = [self.dinov2.embed_dim] * len(self.dinov2.blocks)
            self.output_indexes = list(np.arange(len(self.dinov2.blocks)))
        else:
            self.out_channels = getattr(self.dinov2, "embed_dims", None)
            self.output_indexes = 4

        self.return_cls_token = return_cls_token

    def forward(self, x: Tensor):
        """
        Forward pass for the model.

        Args:
            x: tensor of shape (B, 3, H, W)

        Returns:
            list of embeddings for all the intermediate layers blocks.
            If `return_cls_token` is True, the CLS token is concatenated
            (appended) per layer as the last token in the sequence.
        """
        feats = self.dinov2.get_intermediate_layers(x, n=self.output_indexes, return_class_token=self.return_cls_token)

        if self.return_cls_token:
            feats = [torch.cat([f[0], f[1].unsqueeze(1)], dim=1) for f in feats]

        return list(feats)


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov2_vits14(ckpt_path: str = None, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov2_vits14 model.

    Args:
        ckpt_path: local path to the desired DINO V2 model checkpoint downloaded from Meta
                   (https://ai.meta.com/resources/models-and-libraries/dinov2/)
        return_cls_token: whether the model should return the class token or not.
                          If True, the class token will be concatenated into the sequence.

    Returns:
        DinoV2Wrapper dinov2_vits14 model
    """
    model = DinoV2Wrapper(model="dinov2_vits14", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)
    return model
