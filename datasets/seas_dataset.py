import random
from pathlib import Path
import os

from PIL import Image
from PIL.ImageOps import exif_transpose

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class RandomRotation:
    def __init__(self, angles):
        self.angles = angles
        self.angle = None

    def __call__(self, x):
        if self.angle is None:
            self.angle = random.choice(self.angles)
        return TF.rotate(x, self.angle)

    def reset(self):
        self.angle = None

class RandomFlip:
    def __init__(self):
        self.methods = [None, Image.FLIP_LEFT_RIGHT , Image.FLIP_TOP_BOTTOM]
        self.method = None
        self.new = True

    def __call__(self, x):
        if self.new is True:
            self.method = random.choice(self.methods)
            self.new = False
        if self.method is None:
            x = x
        elif self.method is not None:
            x = x.transpose(self.method)
        return x

    def reset(self):
        self.new = True

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    """
    tokenize_prompt use the tokenizer to tokenize the prompt, and return the token ids.
    """
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs

class SeaSTrainDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        instance_data_root,
        mask_root,
        tokenizer,
        normal_data_root=None,
        normal_num=None,
        size=512,
        center_crop=True,
        rotation=False,
        tokenizer_max_length=None,
        normal_token_num=None, # num of normal tokens
        anomaly_token_num=None, # num of anomlay tokens
        pretrained_model_name_or_path=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.rotation = rotation
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.normal_token_num = normal_token_num
        self.anomaly_token_num = anomaly_token_num
        self.pretrained_model_name_or_path = pretrained_model_name_or_path,

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")
        
        self.mask_root = Path(mask_root)
        if not self.mask_root.exists():
            raise ValueError(f"Mask {self.mask_root}  root doesn't exists.")
        self.instance_images_path = list(self.instance_data_root.iterdir()) 
        self.mask_path = list(self.mask_root.iterdir())

        self.num_instance_images_list = {}
        for subfolder in sorted(Path(instance_data_root).iterdir()):
            if subfolder.is_dir() and subfolder.name != "good":
                num_images = len(list(subfolder.iterdir()))
                self.num_instance_images_list[subfolder.name] = int(num_images/3)
        self.total_num_images = sum(self.num_instance_images_list.values())

        # cumulative_lengths is used to store the cumulative length of images of different anomlay types.
        self.cumulative_lengths = [0] 
        for name, length in self.num_instance_images_list.items():
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

        self._length = self.total_num_images
        
        self.rotation_transform = RandomRotation([0, 90, 180, 270])
        self.flip_transform = RandomFlip()

        if normal_data_root is not None:
            self.normal_data_root = Path(normal_data_root)
            self.normal_data_root.mkdir(parents=True, exist_ok=True)
            self.normal_images_path = list(self.normal_data_root.iterdir())
            if normal_num is not None:
                self.num_normal_images = min(len(self.normal_images_path), normal_num)
            else:
                self.num_normal_images = len(self.normal_images_path)
            self._length = max(self.num_normal_images, self.total_num_images)
        else:
            self.normal_data_root = None

        if rotation == True:
            self.image_transforms = transforms.Compose(
                [
                    self.rotation_transform,
                    self.flip_transform,
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.mask_transform = transforms.Compose(
                [
                    self.rotation_transform,
                    self.flip_transform,
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        cumulative_lengths = self.cumulative_lengths
        index = index % self.total_num_images
        defect_name = list(self.num_instance_images_list.keys())[-1] # default to last anomaly category
        # Mixed Training: Loop through cumulative lengths to determine which anomaly category the index belongs to.
        # We random select a anomaly image from all the anomaly iameges using index,
        # then use the index and cumulative_lengths to judge which anomaly category this image belongs to.
        for i in range(len(cumulative_lengths) - 1):
            if index < cumulative_lengths[i+1]:
                defect_index = i
                defect_name = list(self.num_instance_images_list.keys())[i] # Calculate local index within the anomaly category.
                break
        index_defect = index - cumulative_lengths[i]
        image_file_name = f"{index_defect:03}.png"
        image_path = os.path.join(self.instance_data_root,defect_name,image_file_name)
        instance_image = Image.open(image_path)
        mask_file_name = f"{index_defect:03}_mask.png"
        if "visa" in str(self.instance_data_root):
            mask_file_name = f"{index_defect:03}.png"
        mask_path = os.path.join(self.mask_root,defect_name,mask_file_name)
        mask = Image.open(mask_path)
        instance_image = exif_transpose(instance_image)
        mask = exif_transpose(mask)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["mask"] = self.mask_transform(mask)
        example["label"] = defect_index

        # Unbalanced Abnormal Prompt 
        # We form the prompt acording to the anomaly category index.
        # This is to deal with the situation where a batch contains images that belong to different anomaly categories.
        sks_start = example["label"] * self.anomaly_token_num + 1
        sks_tokens = [f"sks{sks_start + i}" for i in range(self.anomaly_token_num)]
        ob_tokens = [f"ob{i}" for i in range(1, self.normal_token_num + 1)]
        unbalanced_prompt = f"a {' '.join(ob_tokens)} with {' '.join(sks_tokens)}"

        text_inputs = tokenize_prompt(
            self.tokenizer, unbalanced_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["unbalanced_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        # For normal images
        if self.normal_data_root:
            normal_prompt = f"a {' '.join(ob_tokens)}"
            normal_image = Image.open(self.normal_images_path[index % self.num_normal_images])
            normal_image = exif_transpose(normal_image)

            if not normal_image.mode == "RGB":
                normal_image = normal_image.convert("RGB")
            example["normal_images"] = self.image_transforms(normal_image)

            normal_text_inputs = tokenize_prompt(
                self.tokenizer, normal_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["normal_prompt_ids"] = normal_text_inputs.input_ids
            example["normal_attention_mask"] = normal_text_inputs.attention_mask

        self.rotation_transform.reset()
        self.flip_transform.reset()

        return example


def collate_fn(examples, with_Ni_Alignment=False):
    """
    The collate_fn is used as the collate_fn in dataloader to orgnize the batch.
    If with_Ni_Alignment is True, we put anomlay images and normal images together into the pixel_values.
    """
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["unbalanced_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    label = [example["label"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
    if with_Ni_Alignment:
        input_ids += [example["normal_prompt_ids"] for example in examples]
        pixel_values += [example["normal_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["normal_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    mask = torch.cat(mask, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "mask": mask,
        "label":label
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask 
    return batch


class SeaSTestDataset(Dataset):
    """
    A dataset to prepare the normal images which are used as initial noise after adding random noise during inference.
    """

    def __init__(
        self,
        instance_data_root,
        size=512,
        center_crop=True,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        return example

def collate_fn_test(examples):
    """
    The collate_fn_test is used as the collate_fn in dataloader to orgnize the batch 
    using the examples returned by SeaSTestDataset during inference.
    """

    pixel_values = [example["instance_images"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
    }

    return batch