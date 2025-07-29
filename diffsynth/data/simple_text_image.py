import torch, os, torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image
from ast import literal_eval
from pycocotools import mask as coco_mask
import numpy as np


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True,
                 random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        metadata = pd.read_csv(os.path.join(dataset_path, "train/metadata.csv"))
        self.path = [os.path.join(dataset_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        self.height = height
        self.width = width
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        text = self.text[data_id]
        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height * scale), round(width * scale)]
        image = torchvision.transforms.functional.resize(image, shape,
                                                         interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)
        return {"text": text, "image": image}

    def __len__(self):
        return self.steps_per_epoch


def gen_entity_mask(bbox, height, width):
    mask = torch.zeros((3, height, width))
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    mask[:, y1:y2, x1:x2] = 255.0
    return mask


class AmodalTextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True,
                 random_flip=False):
        from datasets import load_dataset
        self.steps_per_epoch = steps_per_epoch
        self.dataset = load_dataset(dataset_path, split="train")
        self.num_samples = len(self.dataset)
        self.height = height
        self.width = width
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_processor = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    def __getitem__(self, index):
        data_id = torch.randint(0, self.num_samples, (1,))[0]
        data_id = (data_id + index) % self.num_samples

        item = self.dataset[data_id.item()]
        image = item["image"].convert("RGB")
        # target_height, target_width = self.height, self.width
        # width, height = image.size
        # scale = max(target_width / width, target_height / height)
        # shape = [round(height * scale), round(width * scale)]
        # image = torchvision.transforms.functional.resize(image, shape,
        #                                                  interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)
        annotation = literal_eval(item["annotation"])
        text = annotation['img_meta']['caption']

        height, width = image.shape[1], image.shape[2]

        # for entity in annotation['entities']:
        #     # TODO: resize the bbox and segmentation mask if we add COCO-Amodal
        #     bbox = entity['bbox']
        #     entity_prompt = entity['prompt']
        #     seg_mask = coco_mask.decode(entity['segmentation'])
        #     entity_mask = self.gen_entity_mask(bbox, height, width)

        entity_mask = [gen_entity_mask(entity['bbox'], height, width) for entity in annotation['entities']]
        seg_mask = torch.tensor(
            np.stack([coco_mask.decode(entity['segmentation']) for entity in annotation['entities']]))
        entity_prompt = [entity['prompt'] for entity in annotation['entities']]

        entity_mask = [self.mask_processor(mask).permute((1, 2, 0)) for mask in entity_mask]
        seg_mask = self.mask_processor(seg_mask)

        return {
            "text": text,
            "image": image,
            "entity_mask": entity_mask,
            "entity_prompt": entity_prompt,
            "masks": seg_mask.to(torch.float32),
        }

    def __len__(self):
        return self.steps_per_epoch

class L2IValDataset(torch.utils.data.Dataset):
    def __init__(self):
        import json
        with open("./evaluation_set.json", "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return 6

    def __getitem__(self, index):
        # TODO: remove overlap
        item = self.dataset[index]
        item['entities'] = [e for e in item['entities'] if e['category'] != "overlap"]
        caption = item['img_meta']["captions"]
        height, width = item['img_meta']["height"], item['img_meta']["width"]
        bboxes = [e['bbox'] for e in item['entities']]
        bboxes = [[bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height] for bbox in bboxes]
        entity_promtps = [e['prompt'] for e in item['entities']]
        entity_mask = [gen_entity_mask(box, height, width).permute((1, 2, 0)) for box in bboxes]

        return {
            "text": caption,
            "entity_mask": entity_mask,
            "entity_prompt": entity_promtps,
            'bboxes': torch.tensor(bboxes),
            "categories": [e['category'] for e in item['entities']],
        }
