import glob
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F

from base_dataset import get_params, get_transform


class MultipleDataset(VisionDataset):

    SUBFOLDERS = ("cloudy", "clear")

    def __init__(
        self,
        root: str,
        band: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, target_transform, transforms)

        self.band = band

        cloudy_0_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_0.jpg")
        cloudy_1_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_1.jpg")
        cloudy_2_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_2.jpg")

        cloudy_0_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_0_ir.jpg")
        cloudy_1_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_1_ir.jpg")
        cloudy_2_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_2_ir.jpg")

        clear_pathname = os.path.join(root, self.SUBFOLDERS[1], "*")

        self.cloudy_0_rgb_paths = sorted(glob.glob(cloudy_0_rgb_pathname))
        self.cloudy_1_rgb_paths = sorted(glob.glob(cloudy_1_rgb_pathname))
        self.cloudy_2_rgb_paths = sorted(glob.glob(cloudy_2_rgb_pathname))

        self.cloudy_0_ir_paths = sorted(glob.glob(cloudy_0_ir_pathname))
        self.cloudy_1_ir_paths = sorted(glob.glob(cloudy_1_ir_pathname))
        self.cloudy_2_ir_paths = sorted(glob.glob(cloudy_2_ir_pathname))

        self.clear_paths = sorted(glob.glob(clear_pathname))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        cloudy_0_rgb = Image.open(
            self.cloudy_0_rgb_paths[index]).convert('RGB')
        cloudy_1_rgb = Image.open(
            self.cloudy_1_rgb_paths[index]).convert('RGB')
        cloudy_2_rgb = Image.open(
            self.cloudy_2_rgb_paths[index]).convert('RGB')

        if self.band == 4:
            cloudy_0_ir = Image.open(
                self.cloudy_0_ir_paths[index]).convert('RGB')
            cloudy_1_ir = Image.open(
                self.cloudy_1_ir_paths[index]).convert('RGB')
            cloudy_2_ir = Image.open(
                self.cloudy_2_ir_paths[index]).convert('RGB')
        else:
            pass

        clear = Image.open(self.clear_paths[index]).convert('RGB')

        params = get_params(size=clear.size)
        transform_params = get_transform(self.band, params)

        cloudy_0_rgb_tensor = transform_params(cloudy_0_rgb)
        cloudy_1_rgb_tensor = transform_params(cloudy_1_rgb)
        cloudy_2_rgb_tensor = transform_params(cloudy_2_rgb)
        if self.band == 4:
            cloudy_0_ir_tensor = transform_params(cloudy_0_ir)[:1, ...]
            cloudy_1_ir_tensor = transform_params(cloudy_1_ir)[:1, ...]
            cloudy_2_ir_tensor = transform_params(cloudy_2_ir)[:1, ...]
        else:
            cloudy_0_ir_tensor = None
            cloudy_1_ir_tensor = None
            cloudy_2_ir_tensor = None
        clear_tensor = transform_params(clear)

        cloudy_0 = torch.cat(
            [i for i in [cloudy_0_rgb_tensor, cloudy_0_ir_tensor] if i is not None])
        cloudy_1 = torch.cat(
            [i for i in [cloudy_1_rgb_tensor, cloudy_1_ir_tensor] if i is not None])
        cloudy_2 = torch.cat(
            [i for i in [cloudy_2_rgb_tensor, cloudy_2_ir_tensor] if i is not None])

        clear = clear_tensor

        return [cloudy_0, cloudy_1, cloudy_2], clear, self.clear_paths[index]

    def __len__(self) -> int:
        return len(self.clear_paths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for item in MultipleDataset(root="./data/multipleImage", band=3):
        print(item["cloudy_0"].shape, item["cloudy_0"].dtype)
        print(item["cloudy_1"].shape, item["cloudy_1"].dtype)
        print(item["cloudy_2"].shape, item["cloudy_2"].dtype)
        print(item["clear"].shape, item["clear"].dtype)
        print(item["cloudy_0_path"].replace("\\", "/"))
        print(item["cloudy_1_path"].replace("\\", "/"))
        print(item["cloudy_2_path"].replace("\\", "/"))

        plt.figure(figsize=(8, 32), dpi=300)

        plt.subplot(1, 4, 1)
        plt.title("cloudy_0")
        plt.imshow(item["cloudy_0"][:3, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 4, 2)
        plt.title("cloudy_1")
        plt.imshow(item["cloudy_1"][:3, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 4, 3)
        plt.title("cloudy_2")
        plt.imshow(item["cloudy_2"][:3, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 4, 4)
        plt.title("clear")
        plt.imshow(item["clear"].permute(1, 2, 0)*0.5+0.5)

        plt.savefig("paired.png", bbox_inches="tight")

        break
    for item in MultipleDataset(root="./data/multipleImage", band=4):
        print(item["cloudy_0"].shape, item["cloudy_0"].dtype)
        print(item["cloudy_1"].shape, item["cloudy_1"].dtype)
        print(item["cloudy_2"].shape, item["cloudy_2"].dtype)
        print(item["clear"].shape, item["clear"].dtype)
        print(item["cloudy_0_path"].replace("\\", "/"))
        print(item["cloudy_1_path"].replace("\\", "/"))
        print(item["cloudy_2_path"].replace("\\", "/"))

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 7, 1)
        plt.title("cloudy_0")
        plt.imshow(item["cloudy_0"][:3, ...].permute(1, 2, 0)*0.5+0.5)
        plt.subplot(1, 7, 2)
        plt.title("cloudy_0_ir")
        plt.imshow(item["cloudy_0"][3:, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 7, 3)
        plt.title("cloudy_1")
        plt.imshow(item["cloudy_1"][:3, ...].permute(1, 2, 0)*0.5+0.5)
        plt.subplot(1, 7, 4)
        plt.title("cloudy_1_ir")
        plt.imshow(item["cloudy_1"][3:, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 7, 5)
        plt.title("cloudy_2")
        plt.imshow(item["cloudy_2"][:3, ...].permute(1, 2, 0)*0.5+0.5)
        plt.subplot(1, 7, 6)
        plt.title("cloudy_2_ir")
        plt.imshow(item["cloudy_2"][3:, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 7, 7)
        plt.title("clear")
        plt.imshow(item["clear"].permute(1, 2, 0)*0.5+0.5)

        plt.savefig("paired.png", bbox_inches="tight")

        break
