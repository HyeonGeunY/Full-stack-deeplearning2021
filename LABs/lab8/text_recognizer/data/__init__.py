from .util import BaseDataset
from .base_data_module import BaseDataModule
from .mnist import MNIST
from .emnist import EMNIST
from .emnist_lines import EMNISTLines
from .emnist_lines2 import EMNISTLines2
from .iam import IAM
from .iam_lines import IAMLines
from .iam_paragraphs import IAMParagraphs
from .iam_original_and_synthetic_paragraphs import IAMOriginalAndSyntheticParagraphs

# __iit__.py 파일을 만들면서 해당 디렉토리를 패키지로 만든다.
# __init__.py 파일에 from .util import BaseDataset과 같은 구문을 넣음으로써 중간 모듈 이름을 생략할 수 있다.
# ex)
# import data.util.BaseDatset -> import data
# data.util.BaseDataset -> data.BaseDataset
