import os

import numpy as np
from crackseg.utils.general import Augmentation
from PIL import Image
from torch.utils import data


class RoadCrack(data.Dataset):
    def __init__(
        self, root: str, image_size: int = 448, transforms: Augmentation = Augmentation(), mask_suffix: str = "_mask"
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.root, "images"))]
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        image_path = os.path.join(self.root, f"images{os.sep}{filename}.jpg")
        mask_path = os.path.join(self.root, f"masks{os.sep}{filename + self.mask_suffix}.jpg")

        # image load
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # TODO: The mask must be binary. In `Road Crack` dataset the mask image has values between 0 and 255, however
        #  it was supposed to be 0 and 1. So mask image divided by 255 to make it between 0 and 1.
        if (np.asarray(mask) > 1).any():
            mask = np.asarray(np.asarray(mask) / 255, dtype=np.byte)
            mask = Image.fromarray(mask)
        assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        # resize
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask
