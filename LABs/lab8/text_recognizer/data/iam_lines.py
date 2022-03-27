import sys
sys.path.append('../..')
from pathlib import Path

from typing import Sequence
import argparse
import json
import random

from PIL import Image, ImageFile, ImageOps
import numpy as np
from torchvision import transforms

from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, split_dataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.iam import IAM
from text_recognizer import util

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_lines"
TRAIN_FRAC = 0.8
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 2048

class IAMLines(BaseDataModule):
    """
    IAM Handwriting database lines.
    """
    
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true") == "true"
        self.mapping = EMNIST().mapping
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
        self.output_dims = (89, 1)
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser
    
    def prepare_data(self, *args, **kwargs) -> None:
        if PROCESSED_DATA_DIRNAME.exists():
            return
        
        print("Cropping IAM line regions...")
        iam = IAM()
        iam.prepare_data()
        crops_trainval, labels_trainval = line_crops_and_labels(iam, "trainval")
        crops_test, labels_test = line_crops_and_labels(iam, "test")
        
        shapes = np.array([crop.size for crop in crops_trainval + crops_test]) # crop: PIL.Image, .size => width, height 반환
        aspect_ratios = shapes[:, 0] / shapes[:, 1] # height / width
        
        print("Saving images, labels and statistics...")
        save_images_and_labels(crops_trainval, labels_trainval, "trainval", PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops_test, labels_test, "test", PROCESSED_DATA_DIRNAME)
        with open(PROCESSED_DATA_DIRNAME / "_max_aspect_ratio.txt", "w") as file:
            file.write(str(aspect_ratios.max()))
             # max aspect ratio 저장하는 이유: 데이터 확장 단계에서 전체 배경이미지에 문장 crop을 삽입, 이미지의 크기를 일정하게 하기위해 최대 이미지 크기로 배경을 만든다.
    
    def setup(self, stage: str = None) -> None:
        """처리된 이미지를 받아 훈련에 사용할 수 있도록 만든다.
        """
        with open(PROCESSED_DATA_DIRNAME / "_max_aspect_ratio.txt") as file:
            max_aspect_ratio = float(file.read())
            image_width = int(IMAGE_HEIGHT * max_aspect_ratio)
            assert image_width <= IMAGE_WIDTH
        
        if stage == "fit" or stage is None:
            x_trainval, labels_trainval = load_line_crops_and_labels("trainval", PROCESSED_DATA_DIRNAME)
            assert self.output_dims[0] >= max([len(_) for _ in labels_trainval]) + 2 # +2: start, end 토근을 위한 자리
            
            y_trainval = convert_strings_to_labels(labels_trainval, self.inverse_mapping, length=self.output_dims[0]) 
            data_trainval = BaseDataset(x_trainval, y_trainval, transform=get_transform(IMAGE_WIDTH, self.augment))
            
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)
            
        if stage == "test" or stage is None:
            x_test, labels_test = load_line_crops_and_labels("test", PROCESSED_DATA_DIRNAME)
            assert self.output_dims[0] >= max([len(_) for _ in labels_test]) + 2
            
            y_test = convert_strings_to_labels(labels_test, self.inverse_mapping, length=self.output_dims[0])
            self.data_test = BaseDataset(x_test, y_test, transform=get_transform(IMAGE_WIDTH))

        if stage is None:
            
            self._verify_output_dims(labels_trainval=labels_trainval, labels_test=labels_test)
            
    def _verify_output_dims(self, labels_trainval, labels_test):
        """
        사용할 데이터의 최대 길이가 설정한 아웃풋 사이즈와 일치하는지 확인한다.
        """
        max_label_length = max([len(label) for label in labels_trainval + labels_test]) + 2
        output_dims = (max_label_length, 1)
        if output_dims != self.output_dims:
            raise RuntimeError(output_dims, self.output_dims)
        
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Lines Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
            
        )
        
        return basic + data
    
                    
def line_crops_and_labels(iam: IAM, split: str):
    """
    이미지 내에 존재하는 문자를 문장(line) 단위로 담아서 반환한다.
    모든 문단(paragraph) 데이터를 문장 단위로 나누어 이미지(crops)와 labels(문장 단위 텍스트 리스트를 담은 리스트)
    로 저장하여 반환한다.
    Parameters
    -----------
    split: str
        train, val, test 
    """
    crops = []
    labels = []
    for filename in iam.form_filenames:
        if not iam.split_by_id[filename.stem] == split:
            continue
        image = util.read_image_pil(filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)
        labels += iam.line_strings_by_id[filename.stem]
        crops += [
            image.crop([region[_] for _ in ["x1", "y1", "x2", "y2"]]) for region in iam.line_regions_by_id[filename.stem]
        ]
    
    assert len(crops) == len(labels)
    return crops, labels


def save_images_and_labels(crops: Sequence[Image.Image], labels: Sequence[str], split: str, data_dirname: Path):
    """
    trainval, test 를 나누어서 이미지(crops)와 문장(labels)을 저장한다.
    
    Args:
        crops: 문장(line) 단위 이미지 리스트
        labels:  문장 (텍스트) 리스트
        split: trainval or test
        dirname: 저장할 디렉토리 이름.
    """
    (data_dirname / split).mkdir(parents=True, exist_ok=True)
    
    with open(data_dirname / split / "_labels.json", "w") as f:
        json.dump(labels, f)
    for ind, crop in enumerate(crops):
        crop.save(data_dirname / split / f"{ind}.png")
        
        
def load_line_crops_and_labels(split: str, data_dirname: Path):
    """
    이미지(crop)와 레이블(labels) 데이터를 불러온다.
    로드시 이미지와 레이블이 뒤섞이지 않게 crops을 정렬해준다.
    """
    with open(data_dirname / split / "_labels.json") as file:
        labels = json.load(file)
    
    crop_filenames = sorted((data_dirname / split).glob("*.png"), key=lambda filename: int(Path(filename).stem))
    crops = [util.read_image_pil(filename, grayscale=True) for filename in crop_filenames]
    assert len(crops) == len(labels)
    return crops, labels

def get_transform(image_width, augment=False):
    """Augment with brightness, slight rotation, slant, translation, scale, and Gaussian noise."""
    
    def embed_crop(crop, augment=augment, image_width=image_width):
        # crop is PIL.image of dtype="L" (so values range from 0 -> 255)
        # mode="L" (8-bit pixels, black and white)
        # 문장(line 단위 이므로) 높이는 고정.
        image = Image.new("L", (image_width, IMAGE_HEIGHT))
        
        # resize crop
        crop_width, crop_height = crop.size
        new_crop_height = IMAGE_HEIGHT
        new_crop_width = int(new_crop_height * (crop_width / crop_height))
        if augment:
            new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
            new_crop_width = min(new_crop_width, image_width)
        crop_resized = crop.resize((new_crop_width, new_crop_height), resample=Image.BILINEAR) #interpolation
        
        # Embed in the image
        x = min(28, image_width - new_crop_width)
        y = IMAGE_HEIGHT - new_crop_height # y=0 이 맨위.
         
        image.paste(crop_resized, (x, y)) # (x, y) the upper left corner. 
        
        return image
    
    transforms_list = [transforms.Lambda(embed_crop)]
    if augment:
        transforms_list += [
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(degrees=1, shear=(-30, 20), resample=Image.BILINEAR, fillcolor=0),
        ]
    transforms_list.append(transforms.ToTensor()) # 마지막 텐서 데이터 타입 변경.
    return transforms.Compose(transforms_list)
        

if __name__ == "__main__":
    load_and_print_info(IAMLines)
