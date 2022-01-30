"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present
"""
from pathlib import Path
from typing import Sequence
import json
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import numpy as np
import toml  # toml 형식으로 작성된 파이썬 패키지 다운

from text_recognizer.data.base_data_module import (
    _download_raw_dataset,
    BaseDataModule,
    load_and_print_info,
)
from text_recognizer.data.util import BaseDataset, split_dataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"


class EMNIST(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, args=None):
        super().__init__(args)

        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)} # chr : class
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])
        self.output_dims = (1,)

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            _essentials = json.load(f)

    def setup(self, stage: str = None) -> None:
        """
        학습용(train, val) 데이터와 테스트용 데이터를 불러온다.
        stage 변수를 통해 특정 목적 (fit, test)에 따라서 불러오도록 설정
            아무 설정이 전달되지 않을 경우 모두 가져온다.
        """
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = (
                    f["y_train"][:].squeeze().astype(int)
                )  # 1 로 구성된 dimension 제거, 자료형 int형으로 변환 # 나중에 원래 차원이 뭔지 확인해보기 노션에 적기

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            self.data_train, self.data_val = split_dataset(
                base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42
            )

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)

            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        """[summary]
        print 함수로 호출했을 때 반환하는 내용
        train, val, test 중 하나라도 load 하지 않았을 경우
            class 개수, class 값, input 이미지의 크기 요소 반환

        데이터가 모두 load 되었을 경우
        train, valid, testset의 크기
        batch size, type, 최대, 최소 등의 데이터 관련 값 출력
        Returns:
            str: 데이터 관련 내용
        """
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train), {len(self.data_val)}, {len(self.data_test)}}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )

        return basic + data


def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path):
    """[summary]
    dataset을 받은 후 각 class당 개수의 균형을 맞춰준 후 process data dir에 저장한다.

    Args:
        filename (str): [description]
        dirname (Path): [description]
    """

    print("Unzipping EMNIST")
    curdir = os.getcwd()
    os.chdir(dirname)
    zip_file = zipfile.ZipFile(filename, "r")
    zip_file.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat

    print("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    # 클래스가 담겨있는 emnist_essentials.json 파일의 앞 4개는 토큰이므로 target에는 토큰 개수인 4만큼 더 해주어 클래스 시작 index와 맞춰준다.
    # label에는 각 문자열의 class number 가 담겨 있다.
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS

    if SAMPLE_TO_BALANCE:
        # 클래스에 따른 샘플 수의 불균형을 보완하는 과정
        print("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    print("Saving to HDFS in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        # dtype u1 : 넘파이 자료형
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print("Saving essential dataset parameters to text_recognizer/dataset...")
    # mapping : class number를 아스키 코드와 매핑
    # k : class number (in jason), v : 아스키 코드
    mapping = {
        int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]
    }  # chr(v) 아스키 코드 -> 문자 변환
    characters = _augment_emnist_characters(list(mapping.values()))
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}

    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)  # dump : 메모리에 쓰기, dumps: 파이썬 메모리에 올려놓고 계속 사용

    print("Cleaning up...")
    shutil.rmtree("matlab")  # 지정된 폴더와 하위 디렉토리 폴더, 파일를 모두 삭제
    os.chdir(curdir)  # 다시 원래 디렉토리로 돌아오기


def _sample_to_balance(x, y):
    """
    각 클래스별 데이터의 수의 균형이 맞지 않으므로 클래스당 최대 개수를 클래스당 데이터 개수의 평균 만큼 추출한 후 index 반환
    """
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())  # 각 클래스별 개수로 이루어진 array 반환 -> 평균 -> int
    all_sampled_inds = []

    for label in np.unique(y.flatten()):  # np.unique(np.array) array에 담긴 고유한 값들로 이루어진 array반환
        inds = np.where(y == label)[0]  # label과 같은 값을 갖는 y indices 반환
        # num_to_sample 만큼 중복을 허용한 추출 -> np.unique로 중복인자 제거
        # 클래스당 최대로 사용하는 데이터의 개수를 클래스당 데이터 개수의 평균으로 제한
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)  # 사용할 idx 모음
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """
    extra symbol을 포함하여 mapping 확장
    """
    # Extra characters from the IAM dataset
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # Also add special tokens:
    # - CTC blank token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!

    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]


if __name__ == "__main__":
    # print(DL_DATA_DIRNAME)
    load_and_print_info(EMNIST)
