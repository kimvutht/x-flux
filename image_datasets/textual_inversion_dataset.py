import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

from image_datasets.dataset import image_resize


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionCustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir,
        img_size=512,
        caption_type="json",
        placeholder_token="*",
        learnable_property="object",  # [object, style]
        image_formats=("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"),
        repeats=100,
        image_text_list = []
    ):

        # self.image_paths = [
        #     img
        #     for ext in image_formats
        #     for img in glob.glob(f"{img_dir}/**/*.{ext}", recursive=True)
        # ]
        # self.image_paths.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.placeholder_token = placeholder_token
        self.learnable_property = learnable_property
        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.image_text_list = image_text_list
        self.num_images = len(self.image_text_list)
        self._length = self.num_images * repeats

        # self.cache_images = {}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        img = Image.open(self.image_text_list[idx % self.num_images][0]).convert("RGB")
        img = image_resize(img, self.img_size)
        w, h = img.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        img = img.resize((new_w, new_h))
        img = torch.from_numpy((np.array(img) / 127.5) - 1)
        img = img.permute(2, 0, 1)
        # json_path = (
        #     self.image_paths[idx % self.num_images].split(".")[0]
        #     + "."
        #     + self.caption_type
        # )

        # if os.path.exists(json_path):
        #     if self.caption_type == "json":
        #         prompt = json.load(open(json_path))["caption"]
        #     else:
        #         prompt = open(json_path).read()
        # else:
        #     prompt = random.choice(self.templates).format(self.placeholder_token)
        prompt = self.image_text_list[idx % self.num_images][1].format(self.placeholder_token)
        return img, prompt


def textual_inversion_dataset_loader(train_batch_size, num_workers, **args):
    image_text_list = image_template = [
        ("/home/elicer/x-flux/sample_images/샘플_1.png", "A painting in the style of {}, depicting a cheerful community scene with diverse characters engaging in daily activities, from a mother pushing a stroller to children playing with a soccer ball, all set against a backdrop of a quaint town with lush green hills and trees."),
        ("/home/elicer/x-flux/sample_images/샘플_1.png", "People do various activities in a public space in the style of {}"),
        ("/home/elicer/x-flux/sample_images/샘플_2.png", "A painting in the style of {}, showcasing children joyfully caring for pets through activities like playing with cats, feeding birds, brushing dogs, and washing dogs, all framed in a cozy, grid-like layout with a warm, inviting color palette."),
        ("/home/elicer/x-flux/sample_images/샘플_2.png", "Different pet care activities in the style of {}"),
        ("/home/elicer/x-flux/sample_images/샘플_3.png", "A school scene with students having varied emotions and engaging in various activities in the style of {}"),
        ("/home/elicer/x-flux/sample_images/샘플_3.png", "A painting in the style of {}, capturing a lively schoolyard scene where students like Kevin, Sora, and Giho experience a range of emotions, from biking and daydreaming to studying and chatting, set against a backdrop of a charming school building and lush greenery."),
        ("/home/elicer/x-flux/sample_images/샘플_4.png", "A lively classroom scene in the style of {}"),
        ("/home/elicer/x-flux/sample_images/샘플_4.png", "A painting in the style of {}, depicting a vibrant classroom scene where students like Yuna, Amy, Joe, Eric, and others engage in various activities, from studying and playing badminton to checking out posters for the Cooking Club and Art Club, all set in a cheerful educational environment."),
        ("/home/elicer/x-flux/sample_images/샘플_5.png", "A painting in the style of {}, illustrating a bustling park scene where people of all ages enjoy various activities, from biking and walking to having a picnic and washing hands, all set in a cheerful, green environment with winding paths and cozy buildings."),
        ("/home/elicer/x-flux/sample_images/샘플_5.png", "A vibrant outdoor scene with people having daily activities in the style of {}")
    ]

    dataset = TextualInversionCustomImageDataset(**args, image_text_list=image_text_list)
    return DataLoader(
        dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
    )
