"""GPU augmentation for PyTorch training on CUDA.

    import slangimage.augment as A

    train = A.Compose([
        A.RandomResizedCrop(224),
        A.RandomHorizontalFlip(),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = train(images)   # (B, 3, 224, 224) float16 on the current CUDA stream

``images`` is a ``(B, 3, H, W)`` uint8 CUDA tensor or a list of decoded
images of any sizes. Names, parameters and results follow
``torchvision.transforms.v2`` on float images, but random values are drawn
for every image (``same_on_batch=True`` shares them across the batch).
Each pipeline compiles into a few generated Slang kernels. Add your own ops
with :class:`ColorOp` and :class:`NeighbourhoodOp`. Needs PyTorch with CUDA;
torch is imported only when a pipeline runs. See docs/AUGMENT.md.
"""

from ._base import Choice, ColorOp, NeighbourhoodOp, Uniform
from ._compile import Compose
from .transforms import (
    CenterCrop,
    ColorJitter,
    GaussianBlur,
    Normalize,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomSolarize,
    Resize,
)

__all__ = [
    "CenterCrop",
    "Choice",
    "ColorJitter",
    "ColorOp",
    "Compose",
    "GaussianBlur",
    "NeighbourhoodOp",
    "Normalize",
    "RandomApply",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomSolarize",
    "Resize",
    "Uniform",
]
