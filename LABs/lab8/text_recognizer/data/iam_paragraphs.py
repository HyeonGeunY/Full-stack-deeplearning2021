"""IAM paragraph dataset class"""
import sys

sys.path.append("..\..")
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.iam import IAM
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, split_dataset

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_paragraphs"

# 문장 구분을 위한 특수 토큰
NEW_LINE_TOKEN = "\n"
TRAIN_FRAC = 0.8

# 이미지 크기를 절반으로 축소
IMAGE_SCALE_FACTOR = 2
IMAGE_HEIGHT = 1152 // IMAGE_SCALE_FACTOR
IMAGE_WIDTH = 1280 // IMAGE_SCALE_FACTOR
# 문장 최대 길이 제한
MAX_LABEL_LENGTH = 682


class IAMParagraphs(BaseDataModule):
    """
    IAM Handwriting database paragraphs.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true").lower() == "true"
        # EMNIST()를 초기화 한 후 mapping 정보를 가져온다.
        mapping = EMNIST().mapping
        # mapping은 필수 요소 임으로 존재하지 않을 경우 에러를 발생시킨다.
        assert mapping is not None
        # 문장을 구분하기 위한 특수 토큰을 추가한다.
        self.mapping = [*mapping, NEW_LINE_TOKEN]
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)  # We assert that this is correct in setup()
        # MAX_LaBEL_LENGTH 만큼의 개별 문자 결과 샘플을 반환.
        self.output_dims = (MAX_LABEL_LENGTH, 1)  # We assert that this is correct in setup()

    @staticmethod
    def add_to_argparse(parser):
        # 다른 클래스의 staticmethod를 이용하여 parser가 중복된 argument를 요구하지 않도록 할 수 있다.
        BaseDataModule.add_to_argparse(parser)
        # 데이터 확장을 할 것인지 여부
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        """raw data(toml)에서 다운 받은 데이터 (downloaded)에서 데이터를 받아 훈련에 적합한 상태로 처리한 후 저장한다.(processed_data)"""

        if PROCESSED_DATA_DIRNAME.exists():
            return
        print(
            "IAMParagraphs.prepare_data: Cropping IAM paragraph regions and saving them along with labels..."
        )

        iam = IAM()
        iam.prepare_data()
        # ??
        properties = {}
        for split in ["trainval", "test"]:
            # trainval 과 test set별로 문단이 존재하는 이미지 부분과 문단에 포함된 글자를 받는다.
            crops, labels = get_paragraph_crops_and_labels(iam=iam, split=split)
            save_crops_and_labels(crops=crops, labels=labels, split=split)

            # 데이터마다 이미지 크가, 문장길이, 문단 수를 기록한다.
            # upadte : 딕셔너리 내부 값 수정, 없을시 추가.
            properties.update(
                {
                    id_: {
                        "crop_shape": crops[id_].size[
                            ::-1
                        ],  # PIL 이미지의 .size는 (width, height)이므로 헷갈리지 않게 => height, width로 바꾸어 저장한다.
                        "label_length": len(label),
                        "num_lines": _num_lines(label),  # "/n" 의 개수를 센다.
                    }
                    for id_, label in labels.items()
                }
            )

        with open(PROCESSED_DATA_DIRNAME / "_properties.json", "w") as f:
            json.dump(properties, f, indent=4)

    def setup(self, stage: str = None) -> None:
        """processed 데이터를 로드한 후 훈련, 추론에 적합한 상태로 변환."""

        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            # Dataset 반환
            crops, labels = load_processed_crops_and_labels(split)
            X = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops]
            Y = convert_strings_to_labels(
                strings=labels, mapping=self.inverse_mapping, length=self.output_dims[0]
            )
            transform = get_transform(image_shape=self.dims[1:], augment=augment)  # type: ignore
            return BaseDataset(X, Y, transform=transform)

        print(f"IAMParagraphs.setup({stage}): Loading IAM paragraph regions and lines...")
        validate_input_and_output_dimensions(input_dims=self.dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            data_trainval = _load_dataset(split="trainval", augment=self.augment)
            self.data_train, self.data_val = split_dataset(
                base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42
            )

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", augment=False)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
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


def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """저장해 놓은 property 파일을 기반으로 input, output의 차원을 검증한다."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    # 정해둔 max 사이즈보다 큰 이미지가 있으면 에러처리
    assert (
        input_dims is not None
        and input_dims[1] >= max_image_shape[0]
        and input_dims[2] >= max_image_shape[1]
    )

    # 지정한 최대 길이보다 문장의 길이 + 2 (start, end 토큰)가 길면 에러처리.
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """
    scale factor만큼 이미지 사이즈를 변경한다.

    Args
    -------
    image (Image.Image): PIL이미지 파일
    scale_factor (int): 이미지 축소 비율

    """
    if scale_factor == 1:
        return image

    return image.resize(
        (image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR
    )


def get_paragraph_crops_and_labels(
    iam: IAM, split: str
) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """IAM 데이터에서 문단 부분만 자른 이미지와 포함된 문자 레이블을 찾아 반환한다.

    Args
    -------
    iam (IAM): iam 데이터셋
    split (str): trainval or test
    """

    # 이미지와 레이블을 담을 딕셔너리
    crops = {}
    labels = {}
    # iam.form_filenames: 이미지 파일
    for form_filename in iam.form_filenames:
        id_ = form_filename.stem
        # testset일 경우 스킵
        if not iam.split_by_id[id_] == split:
            continue

        # 이미지 파일을 열고, grayscale로 바꾼후 색을 반전한다.
        # 반전함으로써 => 배경은 검은색, 글자는 흰색 이된다.
        image = Image.open(form_filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)

        # iam 라인 위치 정보에서 문단 위치정보를 추출한다.
        line_regions = iam.line_regions_by_id[id_]
        para_bbox = [
            min([_["x1"] for _ in line_regions]),
            min([_["y1"] for _ in line_regions]),
            max([_["x2"] for _ in line_regions]),
            max([_["y2"] for _ in line_regions]),
        ]

        # 문단에 포함된 글자를 받는다.
        lines = iam.line_strings_by_id[id_]

        # 문단 위치 정보로 이미지를 자른다.
        crops[id_] = image.crop(para_bbox)
        # "\n"을 사이에 삽입하여 모든 라인의 문자정보를 합친다.
        labels[id_] = NEW_LINE_TOKEN.join(lines)

    # 모든 id 마다 이미지와 label이 잘 매치 되었는지 확인한다. => 필수적이므로 다를 경우 에러를 발생시킨다.
    assert len(crops) == len(labels)
    return crops, labels


def save_crops_and_labels(crops: Dict[str, Image.Image], labels: Dict[str, str], split: str):
    """crop 이미지와 label 데이터를 trainval, test로 분류하여 저장한다.
    xml 어노테이션에서 필요한 정보를 추출한 후 => json으로 저장하여 학습에 사용한다.
    """
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

    with open(_labels_filename(split), "w") as f:
        json.dump(labels, f, indent=4)

    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))


def load_processed_crops_and_labels(split: str) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """trainval, test에 맞추어 crop 이미지와 label을 로드해서 반환한다.
    label을 키값으로 정리한 후 처리함으로써 이미지와 label이 뒤섞이는 것을 방지한다.
    """
    with open(_labels_filename(split), "r") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [Image.open(_crop_filename(id_, split)).convert("L") for id_ in sorted_ids]
    ordered_labels = [labels[id_] for id_ in sorted_ids]

    assert len(ordered_crops) == len(ordered_labels)
    return ordered_crops, ordered_labels


def get_transform(image_shape: Tuple[int, int], augment: bool) -> transforms.Compose:
    """이미지 transform 파이프라인을 만든다."""
    if augment:
        transforms_list = [
            transforms.RandomCrop(  # random pad image to image_shape with 0
                size=image_shape, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
            ),
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(
                degrees=1,
                shear=(-10, 10),
                resample=Image.BILINEAR,
            ),
        ]
    else:
        # 이미지 크기에 맞춰 padding을 추가하기 위한 변환
        transforms_list = [transforms.CenterCrop(image_shape)]
    transforms_list.append(transforms.ToTensor())  # 마지막은 훈련을 위한 tensor로 항상 변환시켜준다.
    return transforms.Compose(transforms_list)


def get_dataset_properties() -> dict:
    """데이터셋의 전체적인 property 반환"""

    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        # properties의 key는 id, value는 cropsize, label 길이 등을 담고있는 딕셔너리
        return [_[key] for _ in properties.values()]

    # crop_shapes: height, width
    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]  # width / heigth
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {
            "min": min(_get_property_values("num_lines")),
            "max": max(_get_property_values("num_lines")),
        },
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }


def _labels_filename(split: str) -> Path:
    """저장할 label 파일 이름(경로)를 반환한다."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id_: str, split: str) -> Path:
    """저장할 crop 이미지 파일 이름(경로)를 반환한다."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"


def _num_lines(label: str) -> int:
    """label 내부 문장의 개수(줄바꿈개수 + 1)를 반환한다."""
    return label.count("\n") + 1


if __name__ == "__main__":
    load_and_print_info(IAMParagraphs)
