from typing import Dict, Sequence
from LABs.lab2.text_recognizer.data.base_data_module import BaseDataModule
from LABs.lab2.text_recognizer.data.emnist import ESSENTIALS_FILENAME
from collecitons import defaultdict
from pathlib import Path
import argparse

from torchvision import transforms
import h5py
import numpy as np
import torch

from text_recognizer.data.util import BaseDataset
from text_recognizer.data.base_data_module import BaseDataset, load_and_print_info
from text_recognizer.data.emnist import EMNIST

DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_lines_essenetials.json"

MAX_LENGTH = 32
MIN_OVERLAP = 0
MAX_OVERLAP = 0.33
NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST =2000

class EMNISTLines(BaseDataModule):
    """ EMNIST character로 만들어진 합성 손 글씨
    """
    
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        
        self.max_length = self.args.get("max_length", MAX_LENGTH) # ?
        self.min_overlap = self.args.get("min_overlap", MIN_OVERLAP) #?
        self.max_overlap = self.args.get("max_overlap", MAX_OVERLAP) #?
        self.num_train = self.args.get("num_val", NUM_VAL)
        self.num_val = self.args.get("num_test", NUM_TEST)
        self.with_start_end_tokens = self.args.get("with_start_end_tokens", False)
        
        self.emnist = EMNIST()
        self.mapping = self.emnist.mapping
        self.dims = {
            self.emnist.dims[0],
            self.emnist.dims[1],
            self.emnist.dims[2] * self.max_length # 이미지 가로길이는 max_length배(최대 글자 수)만큼 늘어날 수 있음.
        }
        self.output_dims = (self.max_length, 1) # 글자수 만큼의 class
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max line length in characters.")
        parser.add_argument("--min_overlap", type=float, default=MIN_OVERLAP, help="Min overlap between characters in a line, between 0 and 1")
        parser.add_argument("--max_overlap", type=float, default=MAX_OVERLAP, help="Min overlap between characters in a line, between 0 and 1")
        parser.add_argument("--with_start_end_tokens", acition="store_true", default=False)
        return parser # parser를 반환하는 이유?
    
    @property # ?? 
    def data_filename(self):
        return (DATA_DIRNAME
            / f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_ntv{self.num_val}_nte{self.num_test}_{self.with_start_end_tokens}.h5")
    
    def prepare_data(self, *args, **kwargs) -> None:
        if self.data_filename.exists():
            return
        np.random.seed(42)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")
        
        
        
            
        
