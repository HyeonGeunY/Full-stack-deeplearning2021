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
import toml # toml 형식으로 작성된 파이썬 패키지 다운

from text_recognizer.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, split_dataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = BaseDataModule.data_dirname() / "byclass.h5"
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
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
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
        fit or test에 따라서 동작이 다름.
        """
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)
            
            data_trainval = BaseDataset




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
    # mapping : class index를 아스키 코드와 매핑
    # k : class index (in jason), v : 아스키 코드
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]} # chr(v) 아스키 코드 -> 문자 변환
    characters = _augment_emnist_characters(list(mapping.values()))
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}
    
    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f) #dump : 메모리에 쓰기, dumps: 파이썬 메모리에 올려놓고 계속 사용
    
    print("Cleaning up...")
    shutil.rmtree("matlab") # 지정된 폴더와 하위 디렉토리 폴더, 파일를 모두 삭제
    os.chdir(curdir) # 다시 원래 디렉토리로 돌아오기

def _sample_to_balance(x, y):
    """
    각 클래스별 데이터의 수의 균형이 맞지 않으므로 클래스당 최대 개수를 클래스당 데이터 개수의 평균 만큼 추출한 후 index 반환
    """
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean()) # 각 클래스별 개수로 이루어진 array 반환 -> 평균 -> int
    all_sampled_inds = []
    
    for label in np.unique(y.flatten()): # np.unique(np.array) array에 담긴 고유한 값들로 이루어진 array반환
        inds = np.where(y == label)[0] # label과 같은 값을 갖는 y indices 반환
        # num_to_sample 만큼 중복을 허용한 추출 -> np.unique로 중복인자 제거
        # 클래스당 최대로 사용하는 데이터의 개수를 클래스당 데이터 개수의 평균으로 제한
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds) # 사용할 idx 모음
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
    
    
    

