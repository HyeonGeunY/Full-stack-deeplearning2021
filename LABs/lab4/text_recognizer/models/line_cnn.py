from typing import Any, Dict, Union, Tuple
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Param2D = Union[int, Tuple[int, int]]

CONV_DIM = 32
FC_DIM = 512
WINDOW_WIDTH = 16
WINDOW_STRIDE = 8

class ConvBlock(nn.Module):
    '''
    Simple 3x3 conv with padding size 1 (input size를 똑같이 유지)
    '''
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Param2D = 3, stride: Param2D = 1, padding: Param2D = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        -----------
        x
            of dimensions (B, C, H, W)
        Returns
        --------
        torch.Tensor
            of dimensions (B, C, H, W)
        '''
        c = self.conv(x)
        r = self.relu(c)
        return r

class LineCNN(nn.Module):
    '''
    간단한 CNN 모델을 이용하여 윈도우로 문장을 처리하는 모델, output : 숫자(문자 id)의 나열
    '''

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])

        self.output_length = data_config["output_dims"][0]
        
        _C, H, _W = data_config["input_dims"]
        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        self.convs = nn.Sequential(
            ConvBlock(1, conv_dim),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim, stride=2),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim * 2, stride=2),
            ConvBlock(conv_dim * 2, conv_dim * 2),
            ConvBlock(conv_dim * 2, conv_dim * 4, stride=2),
            ConvBlock(conv_dim * 4, conv_dim * 4),
            ConvBlock(conv_dim * 4, fc_dim, kernel_size=(H // 8, self.WW // 8), stride=(H // 8, self.WS // 8), padding=0),)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights in a better way than default
        """

        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear,}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, 1, H, W) input image
        
        Returns
        --------
        torch.Tensor
            (B, C, S) logits, S : 시퀀스 길이, C : class 수 S
            S : W와 self.window_width로 부터 계산
            C : self.num_classes
        """

        _B, _C, _H, _W = x.shape
        x = self.convs(x) # (B, FC_DIM, 1, Sx)
        x = x.squeeze(2).permute(0, 2, 1) # (B, S, FC_DIM)
        x = F.relu(self.fc1(x)) # -> (B, S, FC_DIM)
        x = self.dropout(x)
        x = self.fc2(x) # (B, S, C)
        x = x.permute(0, 2, 1) # -> (B, C, S)
        if self.limit_output_length:
            x = x[:, :, : self.output_length] # 최대 글자수 만큼 자름 (뒤 부분은 버린다.)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--window_width", type=int, default=WINDOW_WIDTH, help="Width of the window that will slide over the input shape")
        parser.add_argument("--window_stride", type=int, default=WINDOW_STRIDE, help="Stride of the window that will slide over the input image.")
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser
    







        
