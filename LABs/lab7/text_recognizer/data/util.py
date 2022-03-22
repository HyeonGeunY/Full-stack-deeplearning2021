"""Base Dataset class."""

from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch

SequenceOrTensor = Union[Sequence, torch.Tensor]

class BaseDataset(torch.utils.data.Dataset):
    """
    Parameters
    -----------
    data
        일반적으로 torch tensors, numpy array, or PIL Images
    targets
        일반적으로 torch tensors or numpy arrays
    transform
        data를 받고 data를 반환하는 함수
    target_transform
        target를 받고 target를 반환하는 함수
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:

        # 훈련 데이터의 샘플의 수 와 타겟의 수가 같은 지 확인
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")

        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        data 길이 반환
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        transform 진행한 데이터 반환

        Parameters
        -----------
        index

        Returns
        --------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        # transform을 딕셔너리에 담고 transform phase를 만드는 것으로 변경 가능
        if self.transform is not None:
            # 전처리
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    
    """
    len(strings)와 length 차이 => len(strings) : batch size, length: 문장 최대 길이
    데이터에 존재하는 문자를 삽입하고 남은 자리는 모두 <P> 토큰을 넣는다.

    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token. 
    """
    
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]

    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        # * 리스트 언패킹 if tokens = ["s", "t", "r", "i", "n", "g"]
        # ["<S>", *tokens, "<E>"]
        # >> ["<S>", "s", "t", "r", "i", "n", "g", "<E>"]
        # ["<S>", *tokens, "<E>"]
        # >> ["<S>", ["s", "t", "r", "i", "n", "g"], "<E>"]

        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]

    return labels


def split_dataset(
    base_dataset: BaseDataset, fraction: float, seed: int
) -> Tuple[BaseDataset, BaseDataset]:
    """
    base_dataset을 두개로 나눔
    1. fraction * base_dataset의 크기
    2. (1-fraction) * base_dataset의 크기
    """

    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size

    return torch.utils.data.random_split( 
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )
