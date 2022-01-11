# %%
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import argparse

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from text_recognizer import util
from text_recognizer.data.util import BaseDataset


def load_and_print_info(data_module_class) -> None:
    """
    EMNISTLines 로드 & info 출력
    """
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser) # parser에 값 넣은 후
    args = parser.parse_args() # 반환 받은 args로
    dataset = data_module_class(args) # data_module_class 인스턴스 생성
    dataset.prepare_data()
    dataset.setup()
    print(dataset)
    

def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"{metadata['url']}에서 {filename}으로 raw dataset 다운로드 중")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename) # sha 256 해쉬맵 계산
    if sha256 != metadata["sha256"]:
        raise ValueError("Download 한 파일의 SHA-256이 metadata document의 sha256과 맞지 않음")
    return filename


BATCH_SIZE = 128
NUM_WORKERS = 0

class BaseDataModule(pl.LightningDataModule):
    """=
    Base DataModule.
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) # arg에 있는 변수들을 반환
        # vars() 지역 변수의 리스트를 반환한다. __dict__ 어트리뷰트를 반환한다. (객체의 내부 변수가 저장된 딕셔너리)
        self.batch_size = self.args.get("batch_size", BATCH_SIZE) # dict.get(key, default=None) 딕셔너리 key에 해당하는 value 반환 없을 시 default 값 반환
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        # isinstance(인스턴스, 데이터나 클래스 타입) 두번째 인자로는 튜플 형태로 여러 개가 들어갈 수 있음.
        # 하나라도 만족하면 True 값 반환
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # 아래 인자들이 subclass에 있는 지 확인
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]
        
    @classmethod # 모든 클래스에 적용되는 매소드
    def data_dirname(cls):
        # Path : 파일경로 객체로 다루는 라이브러리, 문자열을 사용하는 os.path 모듈보다 편리, resolve() : 상대경로를 절대 경로로 변환
        # relative_to() 절대 경로를 상대 경로로 변환
        # parents[] 상위 경로로 이동 [] 0~ 한칸 상위
        return Path(__file__).resolve().parents[3] / "data" # 4칸 상위 디렉토리로 이동 __file__ 현재 코드가 담겨있는 파일의 위치
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="한 스탭에 사용할 samples의 개수"
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="데이터 로드에 사용할 process의 개수"
        )
        return parser
    
    def config(self):
        """
        dataset의 중요 세팅들 반환, instantiate models에 전달되는 값
        """
        
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}
        

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        
    def train_dataloader(self):
        return DataLoader(
            self.data_train, # self.data_train 어디?
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
                                                            