import random

import numpy as np
import torch
from torchvision.transforms import functional as F


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Augmentation:
    """Standard Augmentation"""

    def __init__(self, hflip_prop: float = 0.5) -> None:
        transforms = []
        if hflip_prop > 0:
            transforms.append(RandomHorizontalFlip(hflip_prop))
        transforms.extend([PILToTensor(), ConvertImageDtype(torch.float)])
        self.transforms = Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class PILToTensor:
    """Convert PIL image to torch tensor"""

    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    """Convert Image dtype"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Compose:
    """Composing all transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target
