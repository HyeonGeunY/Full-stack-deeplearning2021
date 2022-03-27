"""
Utility functions for text_recognizer module
"""

from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import base64
import hashlib

from PIL import Image
from tqdm import tqdm
import numpy as np
import smart_open  # 스트리밍 방식으로 효율적으로 큰 크기의 이미지 파일을 사용할 수 있게 해줌


def to_categorical(y, num_classes):
    """
    원-핫 인코딩 텐서 반환
    """
    return np.eye(num_classes, dtype="uint8")[y]


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image


def compute_sha256(filename: Union[Path, str]):
    """
    Return SHA256 checksum of a file
    """
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """
    From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        blocks: int, optional
            지금까지 전달 받은 블록의 개수 [default: 1]
        bsize: int, optional
            각 블록의 크기 (in tqdm units) [default: 1]
        tsize: int, optional
            전체 크기 (in tqdm units). If [default: None] remain unchanged
        """

        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download_url(url, filename):
    """
    url로 부터 filename으로 파일 다운로드, with progress bar
    """

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)
