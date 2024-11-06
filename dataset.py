import os
import wget
from typing import Any
import cv2
import torch
import re
import json
import random
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


PAD_IMG = "./vwp_pad_img.jpg"


# This Simple Dataset is just for testing the pipeline works well.
# Note: You should write a DataLoader suitable for your own Dataset!!!
class SimpleDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(self.root, "image")
        self.mask_dir = os.path.join(self.root, "mask")
        self.text_dir = os.path.join(self.root, "text")

        folders = sorted(os.listdir(self.image_dir))
        self.image_list = [os.path.join(self.image_dir, file) for file in folders]
        self.mask_list = [os.path.join(self.mask_dir, file) for file in folders]
        self.text_list = [
            os.path.join(self.text_dir, file.replace(".png", ".txt"))
            for file in folders
        ]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        text = self.text_list[index]

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")
        image = image.resize((512, 512))
        ref_image = image.resize((224, 224))
        mask = mask.resize((512, 512))

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        ref_image = transforms.ToTensor()(ref_image)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        ref_image = torch.from_numpy(np.ascontiguousarray(ref_image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()[
            [0], :, :
        ]  # 1 channel is enough
        # normalize
        image = image * 2.0 - 1.0
        ref_image = ref_image * 2.0 - 1.0

        with open(text, "r") as f:
            prompt = f.read()

        return {"image": image, "ref_image": ref_image, "mask": mask, "prompt": prompt}


class StorySalonDataset(Dataset):
    def __init__(self, root, dataset_name):
        self.root = root
        self.dataset_name = dataset_name

        self.train_image_list = []
        self.train_mask_list = []
        self.train_text_list = []
        self.test_image_list = []
        self.test_mask_list = []
        self.test_text_list = []

        self.PDF_test_set = []
        self.video_test_set = []
        for line in open(os.path.join(self.root, "PDF_test_set.txt")).readlines():
            self.PDF_test_set.append(line[:-1])
        for line in open(os.path.join(self.root, "video_test_set.txt")).readlines():
            self.video_test_set.append(line[:-1])

        keys = ["African", "Bloom", "Book", "Digital", "Literacy", "StoryWeaver"]
        for key in keys:
            self.PDF_image_dir = os.path.join(self.root, "Image_inpainted", key)
            self.PDF_mask_dir = os.path.join(self.root, "Mask", key)
            self.PDF_text_dir = os.path.join(self.root, "Text", "Caption", key)
            PDF_folders = sorted(os.listdir(self.PDF_image_dir))  # 000575

            self.train_image_folders = [
                os.path.join(self.PDF_image_dir, folder)
                for folder in PDF_folders
                if folder not in self.PDF_test_set
            ]
            self.train_mask_folders = [
                os.path.join(self.PDF_mask_dir, folder)
                for folder in PDF_folders
                if folder not in self.PDF_test_set
            ]
            self.train_text_folders = [
                os.path.join(self.PDF_text_dir, folder)
                for folder in PDF_folders
                if folder not in self.PDF_test_set
            ]
            self.test_image_folders = [
                os.path.join(self.PDF_image_dir, folder)
                for folder in PDF_folders
                if folder in self.PDF_test_set
            ]
            self.test_mask_folders = [
                os.path.join(self.PDF_mask_dir, folder)
                for folder in PDF_folders
                if folder in self.PDF_test_set
            ]
            self.test_text_folders = [
                os.path.join(self.PDF_text_dir, folder)
                for folder in PDF_folders
                if folder in self.PDF_test_set
            ]

            for (
                video
            ) in self.train_image_folders:  # video: image_folder, /dataset/image/00001
                images = sorted(os.listdir(video))
                if len(images) <= 3:
                    print(video)
                    continue
                else:
                    for i in range(len(images) - 3):
                        self.train_image_list.append(
                            [
                                os.path.join(video, images[i]),
                                os.path.join(video, images[i + 1]),
                                os.path.join(video, images[i + 2]),
                                os.path.join(video, images[i + 3]),
                            ]
                        )

            for (
                video
            ) in self.train_mask_folders:  # video: mask_folder, /dataset/mask/00001
                masks = sorted(os.listdir(video))
                if len(masks) <= 3:
                    continue
                else:
                    for i in range(len(masks) - 3):
                        self.train_mask_list.append(
                            [
                                os.path.join(video, masks[i]),
                                os.path.join(video, masks[i + 1]),
                                os.path.join(video, masks[i + 2]),
                                os.path.join(video, masks[i + 3]),
                            ]
                        )

            for (
                video
            ) in self.train_text_folders:  # video: image_folder, /dataset/image/00001
                texts = sorted(os.listdir(video))
                if len(texts) <= 3:
                    continue
                else:
                    for i in range(len(texts) - 3):
                        self.train_text_list.append(
                            [
                                os.path.join(video, texts[i]),
                                os.path.join(video, texts[i + 1]),
                                os.path.join(video, texts[i + 2]),
                                os.path.join(video, texts[i + 3]),
                            ]
                        )

            for (
                video
            ) in self.test_image_folders:  # video: image_folder, /dataset/image/00001
                images = sorted(os.listdir(video))
                if len(images) <= 3:
                    print(video)
                    continue
                else:
                    for i in range(len(images) - 3):
                        self.test_image_list.append(
                            [
                                os.path.join(video, images[i]),
                                os.path.join(video, images[i + 1]),
                                os.path.join(video, images[i + 2]),
                                os.path.join(video, images[i + 3]),
                            ]
                        )

            for (
                video
            ) in self.test_mask_folders:  # video: mask_folder, /dataset/mask/00001
                masks = sorted(os.listdir(video))
                if len(masks) <= 3:
                    continue
                else:
                    for i in range(len(masks) - 3):
                        self.test_mask_list.append(
                            [
                                os.path.join(video, masks[i]),
                                os.path.join(video, masks[i + 1]),
                                os.path.join(video, masks[i + 2]),
                                os.path.join(video, masks[i + 3]),
                            ]
                        )

            for (
                video
            ) in self.test_text_folders:  # video: image_folder, /dataset/image/00001
                texts = sorted(os.listdir(video))
                if len(texts) <= 3:
                    continue
                else:
                    for i in range(len(texts) - 3):
                        self.test_text_list.append(
                            [
                                os.path.join(video, texts[i]),
                                os.path.join(video, texts[i + 1]),
                                os.path.join(video, texts[i + 2]),
                                os.path.join(video, texts[i + 3]),
                            ]
                        )

        self.video_image_dir = os.path.join(
            "./StorySalon/", "image_inpainted_finally_checked"
        )
        self.video_mask_dir = os.path.join("./StorySalon/", "mask")
        self.video_text_dir = os.path.join(self.root, "Text", "Caption", "Video")
        video_folders = sorted(os.listdir(self.video_image_dir))  # 00001
        self.train_image_folders = [
            os.path.join(self.video_image_dir, folder)
            for folder in video_folders
            if folder not in self.video_test_set
        ]
        self.train_mask_folders = [
            os.path.join(self.video_mask_dir, folder)
            for folder in video_folders
            if folder not in self.video_test_set
        ]
        self.train_text_folders = [
            os.path.join(self.video_text_dir, folder)
            for folder in video_folders
            if folder not in self.video_test_set
        ]
        self.test_image_folders = [
            os.path.join(self.video_image_dir, folder)
            for folder in video_folders
            if folder in self.video_test_set
        ]
        self.test_mask_folders = [
            os.path.join(self.video_mask_dir, folder)
            for folder in video_folders
            if folder in self.video_test_set
        ]
        self.test_text_folders = [
            os.path.join(self.video_text_dir, folder)
            for folder in video_folders
            if folder in self.video_test_set
        ]

        fns = lambda s: sum(
            ((s, int(n)) for s, n in re.findall("(\D+)(\d+)", "a%s0" % s)), ()
        )

        for (
            video
        ) in self.train_image_folders:  # video: image_folder, /dataset/image/00001
            images = sorted(os.listdir(video), key=fns)
            if len(images) <= 3:
                print(video)  # All stories shorter than 4 are in the train set.
                continue
            else:
                for i in range(len(images) - 3):
                    self.train_image_list.append(
                        [
                            os.path.join(video, images[i]),
                            os.path.join(video, images[i + 1]),
                            os.path.join(video, images[i + 2]),
                            os.path.join(video, images[i + 3]),
                        ]
                    )

        for video in self.train_mask_folders:  # video: mask_folder, /dataset/mask/00001
            masks = sorted(os.listdir(video), key=fns)
            if len(masks) <= 3:
                continue
            else:
                for i in range(len(masks) - 3):
                    self.train_mask_list.append(
                        [
                            os.path.join(video, masks[i]),
                            os.path.join(video, masks[i + 1]),
                            os.path.join(video, masks[i + 2]),
                            os.path.join(video, masks[i + 3]),
                        ]
                    )

        for (
            video
        ) in self.train_text_folders:  # video: image_folder, /dataset/image/00001
            texts = sorted(os.listdir(video), key=fns)
            if len(texts) <= 3:
                continue
            else:
                for i in range(len(texts) - 3):
                    self.train_text_list.append(
                        [
                            os.path.join(video, texts[i]),
                            os.path.join(video, texts[i + 1]),
                            os.path.join(video, texts[i + 2]),
                            os.path.join(video, texts[i + 3]),
                        ]
                    )

        for (
            video
        ) in self.test_image_folders:  # video: image_folder, /dataset/image/00001
            images = sorted(os.listdir(video), key=fns)
            if len(images) <= 3:
                print(video)
                continue
            else:
                for i in range(len(images) - 3):
                    self.test_image_list.append(
                        [
                            os.path.join(video, images[i]),
                            os.path.join(video, images[i + 1]),
                            os.path.join(video, images[i + 2]),
                            os.path.join(video, images[i + 3]),
                        ]
                    )

        for video in self.test_mask_folders:  # video: mask_folder, /dataset/mask/00001
            masks = sorted(os.listdir(video), key=fns)
            if len(masks) <= 3:
                continue
            else:
                for i in range(len(masks) - 3):
                    self.test_mask_list.append(
                        [
                            os.path.join(video, masks[i]),
                            os.path.join(video, masks[i + 1]),
                            os.path.join(video, masks[i + 2]),
                            os.path.join(video, masks[i + 3]),
                        ]
                    )

        for (
            video
        ) in self.test_text_folders:  # video: image_folder, /dataset/image/00001
            texts = sorted(os.listdir(video), key=fns)
            if len(texts) <= 3:
                continue
            else:
                for i in range(len(texts) - 3):
                    self.test_text_list.append(
                        [
                            os.path.join(video, texts[i]),
                            os.path.join(video, texts[i + 1]),
                            os.path.join(video, texts[i + 2]),
                            os.path.join(video, texts[i + 3]),
                        ]
                    )

        # In-house data
        # self.pdf_image_dir = os.path.join("/data/home/haoningwu/Dataset/StorySalon/", 'StoryBook_finally_checked', 'image_inpainted_finally_checked')
        # self.pdf_mask_dir = os.path.join("/data/home/haoningwu/Dataset/StorySalon/", 'StoryBook_finally_checked', 'mask')
        # self.pdf_text_dir = os.path.join(self.root, 'Text', 'Caption_new', 'eBook')
        # pdf_folders = sorted(os.listdir(self.pdf_image_dir)) # 00001
        # self.pdf_image_folders = [os.path.join(self.pdf_image_dir, folder) for folder in pdf_folders]
        # self.pdf_mask_folders = [os.path.join(self.pdf_mask_dir, folder) for folder in pdf_folders]
        # self.pdf_text_folders = [os.path.join(self.pdf_text_dir, folder) for folder in pdf_folders]
        # fns = lambda s: sum(((s,int(n))for s, n in re.findall('(\D+)(\d+)','a%s0'%s)),())

        # for video in self.pdf_image_folders: # video: image_folder, /dataset/image/00001
        #     images = sorted(os.listdir(video), key=fns)
        #     if len(images) <= 3:
        #         print(video)
        #         continue
        #     else:
        #         for i in range(len(images) - 3):
        #             self.train_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i+1]), os.path.join(video, images[i+2]), os.path.join(video, images[i+3])])

        # for video in self.pdf_mask_folders: # video: mask_folder, /dataset/mask/00001
        #     masks = sorted(os.listdir(video), key=fns)
        #     if len(masks) <= 3:
        #         continue
        #     else:
        #         for i in range(len(masks) - 3):
        #             self.train_mask_list.append([os.path.join(video, masks[i]), os.path.join(video, masks[i+1]), os.path.join(video, masks[i+2]), os.path.join(video, masks[i+3])])

        # for video in self.pdf_text_folders: # video: image_folder, /dataset/image/00001
        #     texts = sorted(os.listdir(video), key=fns)
        #     if len(texts) <= 3:
        #         continue
        #     else:
        #         for i in range(len(texts) - 3):
        #             self.train_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i+1]), os.path.join(video, texts[i+2]), os.path.join(video, texts[i+3])])

        if self.dataset_name == "train":
            self.image_list = self.train_image_list
            self.mask_list = self.train_mask_list
            self.text_list = self.train_text_list
        elif self.dataset_name == "test":
            self.image_list = self.test_image_list
            self.mask_list = self.test_mask_list
            self.text_list = self.test_text_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        ref_image_ids = self.image_list[index][0:3]
        image = self.image_list[index][3]
        mask = self.mask_list[index][3]

        ref_texts = self.text_list[index][0:3]
        text = self.text_list[index][3]

        ref_images_0 = []
        for id in ref_image_ids:
            ref_images_0.append(Image.open(id).convert("RGB"))
        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        ref_images_1 = []
        for ref_image in ref_images_0:
            ref_images_1.append(ref_image.resize((512, 512)))
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        ref_images_2 = []
        for ref_image in ref_images_1:
            ref_images_2.append(np.ascontiguousarray(transforms.ToTensor()(ref_image)))
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        ref_prompts = []
        for ref_text in ref_texts:
            with open(ref_text, "r") as f:
                ref_prompts.append(f.read())
        with open(text, "r") as f:
            prompt = f.read()

        # Unconditional generation for classifier-free guidance
        if self.dataset_name == "train":
            p = random.uniform(0, 1)
            if p < 0.05:
                prompt = ""
            p = random.uniform(0, 1)
            if p < 0.1:
                ref_prompts = ["", "", ""]
                ref_images = ref_images * 0.0

        # normalize
        for ref_image in ref_images:
            ref_image = ref_image * 2.0 - 1.0
        # ref_images = ref_images * 2. - 1.
        image = image * 2.0 - 1.0

        return {
            "ref_image": ref_images,
            "image": image,
            "mask": mask,
            "ref_prompt": ref_prompts,
            "prompt": prompt,
        }


class COCOMultiSegDataset(Dataset):
    def __init__(self, root):
        self.seg_path = os.path.join(root, "annotations/instances_train2017.json")
        self.cap_path = os.path.join(root, "annotations/captions_train2017.json")
        self.image_path = os.path.join(root, "train2017")

        with open(self.seg_path, "r") as f:
            seg_data = json.load(f)
        with open(self.cap_path, "r") as f:
            cap_data = json.load(f)

        self.image_list = seg_data["images"]
        self.annotation_list = seg_data["annotations"]
        self.category_list = seg_data["categories"]
        self.caption_list = cap_data["annotations"]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_id = self.image_list[index]["id"]

        image_info = next(image for image in self.image_list if image["id"] == image_id)
        image_path = os.path.join(self.image_path, image_info["file_name"])
        image = np.ascontiguousarray(Image.open(image_path).convert("RGB"))

        masks = [ann for ann in self.annotation_list if ann["image_id"] == image_id]

        captions = [
            item["caption"]
            for item in self.caption_list
            if item["image_id"] == image_id
        ]
        tmp_ref_captions = []

        tmp_ref_images = []  # len(ref_captions) = len(ref_images)

        for annotation in masks:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            segmentation = annotation["segmentation"]
            mask_cat = [
                item["name"]
                for item in self.category_list
                if item["id"] == annotation["category_id"]
            ]
            tmp_ref_captions.append(mask_cat[0])

            for segment in segmentation:
                if len(segment) > 1:
                    poly = np.array(segment)
                    if poly.shape != ():
                        poly = poly.reshape((len(poly) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], color=255)

            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))

        while len(tmp_ref_images) < 3:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))
            tmp_ref_captions.append("")

        if len(tmp_ref_images) > 3:
            ref_images = tmp_ref_images[0:2]
            ref_captions = tmp_ref_captions[0:3]
            for i in range(3, len(tmp_ref_images)):
                tmp_ref_images[2] += tmp_ref_images[i]
            ref_images.append(tmp_ref_images[2])
        else:
            ref_images = tmp_ref_images
            ref_captions = tmp_ref_captions

        ref_images_0 = []
        for id in ref_images:
            ref_images_0.append(Image.fromarray(id).convert("RGB"))
        image = Image.fromarray(image).convert("RGB")

        ref_images_1 = []
        for ref_image in ref_images_0:
            ref_images_1.append(ref_image.resize((512, 512)))
        image = image.resize((512, 512))

        transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(-30, 30), translate=(0.2, 0.2), scale=(0.8, 1.3)
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        ref_images_2 = []
        for ref_image in ref_images_1:
            ref_images_2.append(np.ascontiguousarray(transform(ref_image)))
        image = transforms.ToTensor()(image)

        ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        for ref_image in ref_images:
            ref_image = ref_image * 2.0 - 1.0
        # ref_images = ref_images * 2. - 1.
        image = image * 2.0 - 1.0

        if len(captions) > 0:
            text = captions[random.randint(0, len(captions) - 1)]
        else:
            text = ""

        # Unconditional generation for classifier-free guidance
        p = random.uniform(0, 1)
        if p < 0.05:
            text = ""
        p = random.uniform(0, 1)
        if p < 0.1:
            ref_captions = ["", "", ""]
            ref_images = ref_images * 0.0

        return {
            "image": image,
            "prompt": text,
            "ref_image": ref_images,
            "ref_prompt": ref_captions,
        }


class COCOValMultiSegDataset(Dataset):
    def __init__(self, root):
        self.seg_path = os.path.join(root, "annotations/instances_val2017.json")
        with open(self.seg_path, "r") as f:
            seg_data = json.load(f)

        self.annotation_list = seg_data["annotations"]
        self.category_list = seg_data["categories"]

        self.image_path = os.path.join(root, "val2017")
        self.text_path = os.path.join("./COCOVal", "Caption")

        self.image_list = sorted(os.listdir(self.image_path))
        self.caption_list = sorted(os.listdir(self.text_path))

        # self.image_list = self.image_list[3600:3950]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_id = image_name.split(".")[0]
        text_path = os.path.join(self.text_path, (image_id + ".txt"))

        image_path = os.path.join(self.image_path, image_name)
        image = np.ascontiguousarray(Image.open(image_path).convert("RGB"))

        masks = [
            ann
            for ann in self.annotation_list
            if ann["image_id"] == int(image_id.lstrip("0"))
        ]

        tmp_ref_captions = []

        tmp_ref_images = []  # len(ref_captions) = len(ref_images)

        for annotation in masks:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            segmentation = annotation["segmentation"]
            mask_cat = [
                item["name"]
                for item in self.category_list
                if item["id"] == annotation["category_id"]
            ]
            tmp_ref_captions.append(mask_cat[0])

            for segment in segmentation:
                if len(segment) > 1:
                    poly = np.array(segment)
                    if poly.shape != ():
                        poly = poly.reshape((len(poly) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], color=255)

            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))

        while len(tmp_ref_images) < 3:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            tmp_ref_images.append(cv2.bitwise_and(image, image, mask=mask))
            tmp_ref_captions.append("")

        if len(tmp_ref_images) > 3:
            ref_images = tmp_ref_images[0:2]
            ref_captions = tmp_ref_captions[0:3]
            for i in range(3, len(tmp_ref_images)):
                tmp_ref_images[2] += tmp_ref_images[i]
            ref_images.append(tmp_ref_images[2])
        else:
            ref_images = tmp_ref_images
            ref_captions = tmp_ref_captions

        ref_images_0 = []
        for id in ref_images:
            ref_images_0.append(Image.fromarray(id).convert("RGB"))
        image = Image.fromarray(image).convert("RGB")

        ref_images_1 = []
        for ref_image in ref_images_0:
            ref_images_1.append(ref_image.resize((512, 512)))
        image = image.resize((512, 512))

        transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        ref_images_2 = []
        for ref_image in ref_images_1:
            ref_images_2.append(np.ascontiguousarray(transform(ref_image)))
        image = transforms.ToTensor()(image)

        ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        for ref_image in ref_images:
            ref_image = ref_image * 2.0 - 1.0
        # ref_images = ref_images * 2. - 1.
        image = image * 2.0 - 1.0

        with open(text_path, "r") as f:
            text = f.read()

        return {
            "image": image,
            "prompt": text,
            "ref_image": ref_images,
            "ref_prompt": ref_captions,
            "image_path": image_path,
        }


if __name__ == "__main__":
    train_dataset = COCOMultiSegDataset(root="./COCO2017/")

    print(train_dataset.__len__())

    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
    # B C H W
    for i, data in enumerate(train_data):
        print(i)
        print(data["prompt"])
        print(data["ref_prompt"])

        print(data["ref_image"].shape)
        print(data["image"].shape)
        if i > 9:
            break


class VWPDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(VWPDataset, self).__init__()
        self.args = args
        self.root_dir = args.get(args.dataset).root_dir
        self.subset = subset
        if subset == "train":
            with open(
                self.root_dir + "vwp_train_captions_clean_links_llama70b.json", "r"
            ) as f:
                vwp_data = json.load(f)
        elif subset == "val":
            with open(
                self.root_dir + "vwp_val_captions_clean_links_llama70b_100.json", "r"
            ) as f:
                vwp_data = json.load(f)
        elif subset == "test":
            with open(
                self.root_dir + "vwp_val_captions_clean_links_llama70b.json", "r"
            ) as f:
                vwp_data = json.load(f)
                if args.start is not None and args.end is not None:
                    print(
                        "Subsetting test data from {} to {}".format(
                            args.start, args.end
                        )
                    )
                    vwp_data = vwp_data[args.start : args.end]
        else:
            assert subset in ["train", "val", "test"]

        self.annt_ids = []
        self.narratives = []
        self.captions = []
        self.llama_captions = []
        self.cap_links = []
        self.llama_cap_links = []
        self.images = []
        self.links = {
            "key_nar": [],
            "non_key_nar": [],
            "key_cap": [],
            "non_key_cap": [],
        }

        def add_ref_and_cur_frames(
            args, sample, attribute_name, attribute_list
        ) -> None:
            if len(sample[attribute_name]) <= args.num_ref_imgs:
                print(sample["scene_full_id"])
                return
            else:
                if args.num_ref_imgs == 1:
                    for i in range(len(sample[attribute_name])):
                        if i == 0:
                            pad_element = (
                                PAD_IMG if attribute_name == "image_links" else ""
                            )
                            attribute_list.append(
                                [pad_element, sample[attribute_name][i]]
                            )
                        else:
                            attribute_list.append(
                                [
                                    sample[attribute_name][i - 1],
                                    sample[attribute_name][i],
                                ]
                            )
                else:
                    for i in range(len(sample[attribute_name]) - args.num_ref_imgs):
                        attribute_list.append(
                            sample[attribute_name][i : i + args.num_ref_imgs + 1]
                        )

        # def add_four_frames(sample, attribute_name, attribute_list) -> None:
        #     if len(sample[attribute_name]) <= 3:
        #         print(sample["scene_full_id"])
        #         return
        #     for i in range(len(sample[attribute_name]) - 3):
        #         attribute_list.append(sample[attribute_name][i : i + 4])

        # def add_two_frames(sample, attribute_name, attribute_list) -> None:
        #     if len(sample[attribute_name]) <= 1:
        #         print(sample["scene_full_id"])
        #         return
        #     for i in range(len(sample[attribute_name])):
        #         if i == 0:
        #             pad_element = PAD_IMG if attribute_name == "image_links" else ""
        #             attribute_list.append([pad_element, sample[attribute_name][i]])
        #         else:
        #             attribute_list.append(
        #                 [sample[attribute_name][i - 1], sample[attribute_name][i]]
        #             )
        for sample in vwp_data:
            if len(sample["image_links"]) <= args.num_ref_imgs:
                print(sample["scene_full_id"])
                return
            for i in range(len(sample["image_links"]) - args.num_ref_imgs):
                self.annt_ids.append(
                    sample["scene_full_id"]
                    + "_"
                    + str(sample["story_id"])
                    + "_"
                    + str(i)
                )
            # add_two_frames(sample, "narrative", self.narratives)
            # add_two_frames(sample, "image_links", self.images)
            # add_two_frames(sample, "captions", self.captions)

            add_ref_and_cur_frames(self.args, sample, "narrative", self.narratives)
            add_ref_and_cur_frames(self.args, sample, "image_links", self.images)
            add_ref_and_cur_frames(self.args, sample, "captions", self.captions)

            if subset == "test":
                # add_two_frames(sample, "llama31_caps", self.llama_captions)
                # add_two_frames(sample, "captions_links", self.cap_links)
                # add_two_frames(sample, "llama31_cap_links", self.llama_cap_links)

                add_ref_and_cur_frames(
                    self.args, sample, "llama31_caps", self.llama_captions
                )
                add_ref_and_cur_frames(
                    self.args, sample, "captions_links", self.cap_links
                )
                add_ref_and_cur_frames(
                    self.args, sample, "llama31_cap_links", self.llama_cap_links
                )

            link_map = {"nar": "links_to_nar", "cap": "links_between_cap"}
            for ent_type in ["key", "non_key"]:
                for link_type in ["nar", "cap"]:
                    links_set = []
                    if link_type == "cap":
                        links_set.append({})
                    for raw_link in sample[ent_type][link_map[link_type]]:
                        links = {}
                        for match in re.findall(r"\([^,\(\)]+, [^,\(\)]+\)", raw_link):
                            entity1 = match.split(", ")[0].strip("( )")
                            entity2 = match.split(", ")[1].strip("( )")
                            if link_type == "nar":
                                links[entity1] = entity2  # caption --> narrative
                            else:
                                links[entity2] = entity1  # current --> previous caption
                        links_set.append(links)

                    if self.args.num_ref_imgs == 1:
                        for i in range(len(links_set)):
                            if i == 0:
                                self.links[ent_type + "_" + link_type].append(
                                    [{}, links_set[i]]
                                )
                            else:
                                self.links[ent_type + "_" + link_type].append(
                                    [links_set[i - 1], links_set[i]]
                                )
                    else:
                        for i in range(len(links_set) - self.args.num_ref_imgs):
                            self.links[ent_type + "_" + link_type].append(
                                links_set[i : i + self.args.num_ref_imgs + 1]
                            )

        self.augment = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([args.vwp.img_width, args.vwp.img_height]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # self.augment = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize([224, 512]),
        #         # transforms.RandomAffine(
        #         #     degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)
        #         # ),
        #         # transforms.ColorJitter(
        #         #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        #         # ),
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #     ]
        # )

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )

    # def open_h5(self):
    #     h5 = h5py.File(self.h5_file, "r")
    #     self.h5 = h5[self.subset]

    def _get_caption(self, sid, add_links=False):
        if add_links:
            key_nar_links = self.links["key_nar"][sid]
            captions = self.captions[sid]

            captions_with_links = []
            for cap, link_set in zip(captions, key_nar_links):
                idx = 0
                cap_ls = word_tokenize(cap)

                for cap_ent, nar_ent in link_set.items():
                    cap_ent_ls = word_tokenize(cap_ent)
                    len_ce = len(cap_ent_ls)
                    nar_ent_ls = ["("] + word_tokenize(nar_ent) + [")"]

                    while idx < len(cap_ls) - len_ce + 1:
                        if cap_ls[idx : idx + len_ce] == cap_ent_ls:
                            break
                        idx += 1

                    if idx < len(cap_ls) - len_ce + 1:
                        cap_ls = (
                            cap_ls[: idx + len_ce] + nar_ent_ls + cap_ls[idx + len_ce :]
                        )
                        idx = idx + len_ce + len(nar_ent_ls)

                captions_with_links.append(" ".join(cap_ls))
            return captions_with_links
        else:
            return self.captions[sid]

    def __getitem__(self, index):

        if len(self.cap_links) == 0:  # training mode:
            if "captions" in self.args.out_mode:
                if "links" in self.args.out_mode:
                    image_captions = self._get_caption(index, add_links=True)
                else:
                    image_captions = self._get_caption(index, add_links=False)
        else:  # testing mode
            if "captions" in self.args.out_mode:
                if "llama" in self.args.out_mode:
                    if "links" in self.args.out_mode:
                        image_captions = self.llama_cap_links[index]
                    else:
                        image_captions = self.llama_captions[index]
                else:
                    if "links" in self.args.out_mode:
                        image_captions = self.cap_links[index]
                    else:
                        image_captions = self.captions[index]

        images = []
        texts = []
        for tid, img_link in enumerate(self.images[index]):
            if img_link == PAD_IMG:
                img_file = PAD_IMG
            else:
                out_pth = os.path.join(
                    self.root_dir + "images", img_link.split("/")[-2]
                )
                img_file = os.path.join(out_pth, img_link.split("/")[-1])
                os.makedirs(out_pth, exist_ok=True)
                if not os.path.exists(img_file) and img_file != PAD_IMG:
                    wget.download(img_link, out=out_pth)
            img = np.array(Image.open(img_file).convert("RGB"))
            images.append(img)

            turn_text = f"Plot {str(tid)}: {self.narratives[index][tid]}"
            if "captions" in self.args.out_mode:
                turn_text += f" Caption {str(tid)}: {image_captions[tid]}"
            texts.append(turn_text)

        images = images[1:] if self.args.task == "continuation" else images
        images = (
            torch.stack([self.augment(im) for im in images])
            if self.subset in ["train", "val"]
            else torch.from_numpy(np.array(images))
        )

        ref_images = images[0 : self.args.num_ref_imgs]
        image = images[self.args.num_ref_imgs]

        ref_prompts = texts[0 : self.args.num_ref_imgs]
        prompt = texts[self.args.num_ref_imgs]

        # for ref_image in ref_images:
        #     ref_image = ref_image * 2.0 - 1.0
        # # ref_images = ref_images * 2. - 1.
        # image = image * 2.0 - 1.0

        # Unconditional generation for classifier-free guidance
        if self.subset == "train":
            p = random.uniform(0, 1)
            if p < 0.05:
                prompt = ""
            p = random.uniform(0, 1)
            if p < 0.1:
                ref_prompts = ["" for _ in range(self.args.num_ref_imgs)]
                ref_images = ref_images * 0.0

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        mask = self.augment(mask)

        # from torchvision.utils import save_image

        # for i, im in enumerate(ref_images):
        #     save_image(
        #         im* 2. - 1, f"dataloader_samples/{self.annt_ids[index]}_ref_image_{i}.png"
        #     )
        #     with open(
        #         f"dataloader_samples/{self.annt_ids[index]}_ref_prompt_{i}.txt", "w"
        #     ) as f:
        #         try:
        #             f.write(ref_prompts[i])
        #         except IndexError as e:
        #             print(ref_prompts)
        #             raise IndexError(e)

        # save_image(image* 2. - 1, f"dataloader_samples/{self.annt_ids[index]}_image.png")
        # with open(f"dataloader_samples/{self.annt_ids[index]}_prompt.txt", "w") as f:
        #     f.write(prompt)

        # save_image(mask* 2. - 1, f"dataloader_samples/{self.annt_ids[index]}_mask.png")

        return {
            "sample_id": self.annt_ids[index],
            "ref_image": ref_images,
            "image": image,
            "mask": mask,
            "ref_prompt": ref_prompts,
            "prompt": prompt,
        }

    def __len__(self):
        return len(self.annt_ids)
