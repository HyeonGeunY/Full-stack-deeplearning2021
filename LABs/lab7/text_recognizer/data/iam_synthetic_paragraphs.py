"""IAM Synthetic Paragraphs Dataset class."""
import sys
sys.path.append("../..")
from typing import Any, List, Sequence, Tuple
import random
from PIL import Image
import numpy as np

from text_recognizer.data.iam_paragraphs import (
    IAMParagraphs,
    get_dataset_properties,
    resize_image,
    get_transform,
    NEW_LINE_TOKEN,
    IMAGE_SCALE_FACTOR,
)

from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import line_crops_and_labels, save_images_and_labels, load_line_crops_and_labels
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_synthetic_paragraphs"

class IAMSyntheticParagraphs(IAMParagraphs):
    """
    IAM handwriting 을 기반으로한 합성 문단 데이터
    """
    def prepare_data(self, *args, **kwargs) -> None:
        """
        
        """
        # 중복 작업을 피하기 위한 코드
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("IAMSyntheticParagraphs.prepare_data: preparing IAM lines for synthetic IAM paragraph creation...")
        print("Cropping IAM line regions and loading labels...")
        iam = IAM()
        iam.prepare_data()
        
        # iam 데이터 이미지, label 불러오기
        crops_trainval, labels_trainval = line_crops_and_labels(iam, "trainval")
        crops_test, labels_test = line_crops_and_labels(iam, "test")
        
        # 이미지 리사이즈
        crops_trainval = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops_trainval]
        crops_test = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops_test]
        
        # 이미지, 레이블 저장
        print(f"Saving images and labels at {PROCESSED_DATA_DIRNAME}...")
        save_images_and_labels(crops_trainval, labels_trainval, "trainval", PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops_test, labels_test, "test", PROCESSED_DATA_DIRNAME)
        
        
    def setup(self, stage: str = None) -> None:
        print(f"IAMSyntheticParagraphs.setup({stage}): Loading trainval IAM paragraph regions and lines...")
        
        if stage == "fit" or stage is None:
            line_crops, line_labels = load_line_crops_and_labels("trainval", PROCESSED_DATA_DIRNAME)
            # 문장 데이터를 이용하여 합성 문단 데이터 생성
            X, para_labels = generate_synthetic_paragraphs(line_crops=line_crops, line_labels=line_labels)
            Y = convert_strings_to_labels(strings=para_labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            transform = get_transform(image_shape=self.dims[1:], augment=self.augment)
            self.data_train = BaseDataset(X, Y, transform=transform)
       
    
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Synthetic Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, 0, 0\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data
            
def generate_synthetic_paragraphs(
    line_crops: List[Image.Image], line_labels: List[str], max_batch_size: int = 9) -> Tuple[List[Image.Image], List[str]]:
    """
    개별 문장 데이터를 이용하여 합성 문단 데이터을 생성한 후 반환한다.
    이미지의 크기, 문장의 길이등이 원본 iam 문단 데이터의 최대 값을 넘지 않도록 한다.
    여러 문장 길이를 갖는 문단을 생성한다.
    """
    paragraph_properties = get_dataset_properties()
    
    indices = list(range(len(line_labels)))
    # 합성 데이터의 최대 batchsize는 참고로 하는 iam 데이터의 최대 문장 개수보다 작아야한다. 
    assert max_batch_size < paragraph_properties["num_lines"]["max"]
    
    # 다양한 문장 길이를 갖는 데이터 생성
    batched_indices_list = [[_] for _ in indices]
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size // 2)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=(max_batch_size // 2) + 1, max_batch_size=max_batch_size)
    )
    
    # 각 길이의 문장이 몇 샘플씩 만들어졌는지 출력.
    unique, counts = np.unique([len(_) for _ in batched_indices_list], return_counts=True)
    for batch_len, count in zip(unique, counts):
        print(f"{count} samples with {batch_len} lines")
    
    para_crops, para_labels = [], []
    for para_indices in batched_indices_list:
        para_label = NEW_LINE_TOKEN.join([line_labels[i] for i in para_indices])
        # iam 원본 데이터에 포함된 최대글자 수 보다 길면 스킵.
        if len(para_label) > paragraph_properties["label_length"]["max"]:
            print("Label longer than longest label in original IAM Paragraphs dataset - hence dropping")
            continue
        
        # 문단을 구성하는 개별 문장 이미지를 합쳐서 문단 이미지 생성
        para_crop = join_line_crops_to_form_paragraph([line_crops[i] for i in para_indices])
        max_para_shape = paragraph_properties["crop_shape"]["max"]
        
        # 합성 이미지가 iam 문단 데이터의 최대 이미지 크기보다 큰 경우 스킵.
        if para_crop.height > max_para_shape[0] or para_crop.width > max_para_shape[1]:
            print("Crop larger than largest crop in original IAM Paragraphs dataset - hence dropping")
            continue
        
        para_crops.append(para_crop)
        para_labels.append(para_label)
        
    assert len(para_crops) == len(para_labels)
    return para_crops, para_labels


def join_line_crops_to_form_paragraph(line_crops: Sequence[Image.Image]) -> Image.Image:
    """문장 이미지들을 받아서 하나의 문단 이미지를 만든다."""
    
    # PIL 이미지: width, height -> height, width 전환
    crop_shapes = np.array([_.size[::-1] for _ in line_crops])
    para_height = crop_shapes[:, 0].sum() # 문단의 높이: height의 총합
    para_width = crop_shapes[:, 1].max() # 문단의 너비: 문장 이미지 중 최대 너비
    
    # 전체 문단 이미지 크기를 갖는 검은색 이미지 생성 후 각 문장 이미지 삽입.
    para_image = Image.new(mode="L", size=(para_width, para_height), color=0)
    current_height = 0
    for line_crop in line_crops:
        para_image.paste(line_crop, box=(0, current_height))
        current_height += line_crop.height
        
    return para_image
        
    
def generate_random_batches(values: List[Any], min_batch_size: int, max_batch_size: int) -> List[List[Any]]:
    """
    iam line 데이터셋을 이용하여 iampragraph 합성 데이터를 만든다.
    iam 데이터의 모든 문장을 섞은 후 
    min_batch_size ~ max_batch_size 사이의 문장의 수로 구성된 문단 데이터를 만든 후 반환한다.
    """
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)
    
    start_id = 0
    grouped_values_list = []
    while start_id < len(shuffled_values):
        num_values = random.randint(min_batch_size, max_batch_size)
        grouped_values_list.append(shuffled_values[start_id : start_id + num_values])
        start_id += num_values
        
    assert sum([len(_) for _ in grouped_values_list]) == len(values)
    return grouped_values_list


if __name__ == "__main__":
    load_and_print_info(IAMSyntheticParagraphs)