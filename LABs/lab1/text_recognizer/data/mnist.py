"""
Mnist DataModule
"""

import argparse

from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

# %%
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOAD_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

from six.moves import urllib # 파이썬2, 파이썬3 버전 호환을 위해

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

# %%
class MNIST(BaseDataModule):
    """
    Mnist DataModule.
    base_data_module.py에 pl.LightningDataModule를 상속한 BaseDataModule 클래스를 만들고
    BaseDataModule를 상속하여 MNIST클래스를 만든다.
    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOAD_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 28, 28) # 이미지 사이즈, .size() 매서드를 통해 반환될 값
        self.output_dims = (1,)
        self.mapping = list(range(10))
        
    def prepare_data(self, *args, **kwargs) -> None:
        """
        Train과 test MNIST data를 Pytorch canonical source에서 다운로드.
        """
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage=None) -> None:
        """
        데이터를 받은 후 train, validation, test set으로 분리, dimesion을 정함.
        """
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])
        self.data_test = TorchMNIST(self.data, train=False, transform=self.transform)
        
if __name__ == "__main__":
    load_and_print_info(MNIST)