import os
import pickle

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import center_crop, resize, pil_to_tensor


class FFHQ(Dataset):

    def __init__(self,
                 datadir="FFHQ",
                 size=None,
                 split="all",
                 **kwargs):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, datadir)
        self.image_size = size
        self.split = split
        self._prepare_split()

    def __len__(self):
        return len(self.split_ffhq)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, f"{index:05d}.png")
        #img_path = os.path.join(self.root, "images", f"{index}.png")
        image = Image.open(img_path)
        image = pil_to_tensor(image)
        image = resize(image, self.image_size, antialias=True)
        image = center_crop(image, self.image_size)
        image = image / 127.5 - 1
        image = image.permute(1, 2, 0).contiguous()
        item = {"image": image}

        return item

    def _prepare_split(self):
        if self.split == "train":
            self.split_ffhq = [f"{i}.jpg" for i in range (60000)]
        elif self.split == "test":
            self.split_ffhq = [f"{60000+i}.jpg" for i in range (10000)]
        elif self.split == "all":
            self.split_ffhq = os.listdir(os.path.join(self.root))
        else:
            raise ValueError("split must be one of ('train', 'test', 'all')")
        