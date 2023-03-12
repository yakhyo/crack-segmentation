import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image, ImageOps
from unet.models import UNet


def preprocess(image, is_mask):
    """Preprocess image and mask"""
    img_ndarray = np.asarray(image)
    if not is_mask:
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

    return img_ndarray


def plot_img_and_mask(img, mask):
    """Display image and mask"""
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f"Output mask (class {i + 1})")
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title("Output mask")
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def resize(image, image_size=512):
    """Letter box resizing"""
    w, h = image.size
    scale = min(image_size / w, image_size / h)

    # resize image
    image = image.resize((int(w * scale), int(h * scale)))

    # pad size
    delta_w = image_size - int(w * scale)
    delta_h = image_size - int(h * scale)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # pad image
    image = ImageOps.expand(image, (left, top, right, bottom))

    return image


def predict(model, image, device, image_size=448, conf_thresh=0.5):
    model.eval()
    model.to(device)

    # Preprocess
    image = resize(image, image_size)
    image = torch.from_numpy(preprocess(image, is_mask=False))
    image = image.unsqueeze(0)
    image = image.to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image).cpu()
        if model.out_channels > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > conf_thresh

    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray):
    """Convert mask to image"""
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def parse_opt():
    parser = argparse.ArgumentParser(description="Crack Segmentation inference arguments")
    parser.add_argument("--weights", default="./weights/last.pt", help="Path to weight file (default: last.pt)")
    parser.add_argument("--input", type=str, default="./assets/image.jpg", help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Path to save mask image")
    parser.add_argument("--image-size", type=int, default=448, help="Input image size")
    parser.add_argument("--view", action="store_true", help="Visualize image and mask")
    parser.add_argument("--no-save", action="store_true", help="Do not save the output masks")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold for mask")

    return parser.parse_args()


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(opt.weights):
        ckpt = torch.load(opt.weights, map_location=device)
    else:
        raise AssertionError(f"Trained weights not found in {opt.weights}")
    # Initialize model and load checkpoint
    model = UNet(in_channels=3, out_channels=2)
    model.load_state_dict(ckpt["model"].float().state_dict())

    # Load & Inference
    image = Image.open(opt.input)
    output = predict(model=model, image=image, device=device, image_size=opt.image_size, conf_thresh=opt.conf_thresh)

    # Convert mask to image
    result = mask_to_image(output)
    result.save(opt.output)

    # Visualize
    if opt.view:
        image = resize(image)
        plot_img_and_mask(image, output)


if __name__ == "__main__":
    params = parse_opt()
    main(params)
