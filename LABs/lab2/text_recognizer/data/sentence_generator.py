import itertools
import re
import string
from typing import Optional

import nltk
import numpy as np

from text_recognizer.data.base_data_module import BaseDataModule

NLTK_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "nltk"


class SentenceGenerator:
    """
    Generate text sentence using the Brown corpus
    """

    def __init__(self, max_length: Optional[int] = None):
        self.text = brown_text()
        self.word_starts_inds = [0] + [
            _.start(0) + 1 for _ in re.finditer(" ", self.text)
        ]  # " " 기준 다음 글자를 받음 => 문자열 시작 지점을 받는다. [0] : 첫 문자열
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        """
        brown corpus로 부터 최소 1 ~ max_length 길이 사이의 문자열을 샘플링
        """

        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError("Must provide max_length to this method or when making this object")

        for _ in range(10):  # 에러 출력 전 여러번 시도하도록 설정
            try:
                first_ind = np.random.randint(0, len(self.word_starts_inds) - 1)
                start_ind = self.word_starts_inds[first_ind]
                end_ind_candidates = []
                for ind in range(
                    first_ind + 1, len(self.word_starts_inds)
                ):  # max_length가 될 때 까지 end_ind 리스트에 추가
                    if self.word_starts_inds[ind] - start_ind > max_length:
                        break
                    end_ind_candidates.append(self.word_starts_inds[ind])  # end 후보군 저장
                end_ind = np.random.choice(end_ind_candidates)  # end 후보군 중에서 random choice
                sampled_text = self.text[start_ind:end_ind].strip()  # 좌우 공백 제거하거
                return sampled_text

            except Exception:
                pass

        raise RuntimeError("Was not able to generate a valid string")


def brown_text():
    """
    모든 구두점이 제거된 하나의 문자열을 brown corpus에서 반환
    """
    sents = load_nltk_brown_corpus()  # brown nltk vk파일 읽기
    text = " ".join(itertools.chain.from_iterable(sents))
    text = text.translate({ord(c): None for c in string.punctuation})  # !? 같은 문자 제거
    text = re.sub(" +", " ", text)  # 두개 이상의 공백을 한개로 바꿈
    return text


def load_nltk_brown_corpus():
    """
    load the Brown corpus using the NLTK library.
    """
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()  # 문자열 받기 시도해보고
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)  # 없으면 다운로드
    return nltk.corpus.brown.sents()
