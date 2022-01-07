import itertools
import re
import string 
from typing import Optional

import nltk
import numpy as np

from text_recognizer.data.base_data_module import BaseDataModule

NLTK_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "nltk"

class SentenceGeneretor:
    """
    Generate text sentence using the Brown corpus
    """
    def __init__(self, max_length: Optional[int] = None):
        self.text = brown_text()
        self.word_starts_inds = [0] + [_.start(0) + 1 for _ in re.finditer(" ", self.text)]
        self.max_length = max_length
        
    def generate()    
    
def brown_text():
    """
    모든 구두점이 제거된 하나의 문자열을 brown corpus에서 반환
    """
    sents = load_ntlk_brown_corpus() # brown nltk vk파일 읽기
    text = " ".join(itertools.chain.from_iterable(sents))
    text = text.translate(ord(c): None for c in string.punctuation) # !? 같은 문자 제거
    text = re.sub(" +", " ", text) # 두개 이상의 공백을 한개로 바꿈
    return text


def load_nltk_brown_corpus():
    """
    load the Brown corpus using the NLTK library.
    """
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)
    return nltk.corpus.brown.sents()
        