"""Base Dataset class."""
# In[0]
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch

# In[0]
SequeneOrTensor = Union[Sequence, torch.Tensor]
# In[1]

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
        data: SequeneOrTensor,
        targets: SequeneOrTensor,
        transform: Callable = None,
        target_transform: Callable = None
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
        """data 길이 반환"""
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
        
        
        
        
        
    
        
        
        
