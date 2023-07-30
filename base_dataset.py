import random
import numpy as np
from PIL import Image
from typing import Tuple
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_params(
    size: Tuple[int, int],
    preprocess: str = "resize_and_crop",
    load_size: int = 286,
    crop_size: int = 256,
):
    w, h = size
    new_h = h
    new_w = w

    if preprocess == 'resize_and_crop':
        new_h = new_w = load_size
    elif preprocess == 'scale_width_and_crop':
        new_w = load_size
        new_h = load_size * h // w
    else:
        pass

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(
    band: int,
    params=None,
    preprocess: str = "resize_and_crop",
    load_size: int = 286,
    crop_size: int = 256,
    no_flip: bool = False,
    grayscale=False,
    method=InterpolationMode.BICUBIC,
    convert=True
):
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    else:
        pass

    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    else:
        pass

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], crop_size)))
    else:
        pass

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(
                lambda img: __flip(img, params['flip'])))
    else:
        pass

    if convert:
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ]
    else:
        pass
    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
