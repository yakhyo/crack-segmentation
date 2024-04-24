import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data

from crackseg.utils.general import TrainTransforms


class RoadCrack(data.Dataset):
    def __init__(
            self,
            root: str,
            image_size: int = 448,
            transforms: TrainTransforms = TrainTransforms,
            mask_suffix: str = "_mask"
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.root, "images"))]
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        image_path = os.path.join(self.root, f"images{os.sep}{filename}.jpg")
        mask_path = os.path.join(self.root, f"masks{os.sep}{filename + self.mask_suffix}.jpg")

        # image load
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        mask = to_binary(mask)

        assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        # TODO: letterbox or some other resizing methods should be used if image is not square.
        # resize input
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))

        # transform
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask


def to_binary(mask_image):
    # convert pixels to class indexes
    _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8) // 255
    binary_mask = Image.fromarray(binary_mask)
    return binary_mask
