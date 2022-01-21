from typing import Union

import torch

def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    '''
    x에서 element가 처음 나타나는 인덱스를 반환. 찾을 수 없으면 dim을 따라 x의 길이 반환.

    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    '''

    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim] # element가 처음 나타나는 인덱스가 없다면 길이 반환

    return ind


