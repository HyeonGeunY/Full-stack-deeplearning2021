import argparse
import itertools
import torch

from .base import BaseLitModel
from .metrics import CharacaterErrorRate