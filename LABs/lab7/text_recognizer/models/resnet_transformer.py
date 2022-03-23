import argparse
from typing import Any, Dict
import math
import torch
import torch.nn as nn
import torchvision

from .transformer_util import PositionalEncodingImage, PositionalEncoding, generate_square_subsequent_mask

TF_DIM = 256
TF_FC_DIM = 1024
TF_DROPOUT = 0.4
TF_LAYERS = 4
TF_NHEAD = 4
RESNET_DIM = 512  # hard-coded

class ResnetTransformer(nn.Module):
    """Process the line through a Resnet and process the resulting sequence with a Transformer decoder"""
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.input_dims = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])
        inverse_mapping = {val: ind for ind, val in enumerate(data_config["mapping"])}
        self.start_token = inverse_mapping["<S>"]
        self.end_token = inverse_mapping["<E>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = data_config["output_dims"][0]
        self.args = vars(args) if args is not None else {}
        
        self.dim = self.args.get("tf_dim", TF_DIM) # 분산 벡터의 차원
        tf_fc_dim = self.args.get("tf_fc_dim", TF_FC_DIM)
        tf_nhead = self.args.get("tf_nhead", TF_NHEAD)
        tf_dropout = self.args.get("tf_dropout", TF_DROPOUT)
        tf_layers = self.args.get("tf_layers", TF_LAYERS)
        
        # ## Encoder part - should output  vector sequence of length self.dim per sample
        resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))  # Exclude AvgPool and Linear layers 이미지 데이터는 마지막 부분 레이어를 분산 표현으로 사용한다.
        # Resnet will output (B, RESNET_DIM, _H, _W) logits where _H = input_H // 32, _W = input_W // 32

        # self.encoder_projection = nn.Conv2d(RESNET_DIM, self.dim, kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.encoder_projection = nn.Conv2d(RESNET_DIM, self.dim, kernel_size=1)
        # encoder_projection will output (B, dim, _H, _W) logits
        
        # 인풋의 위치 인코딩 (2D 이미지 데이터)
        self.enc_pos_encoder = PositionalEncodingImage(
            d_model=self.dim, max_h=self.input_dims[1], max_w=self.input_dims[2]
        )  # Max (Ho, Wo)


        # ## Decoder part
        self.embedding = nn.Embedding(self.num_classes, self.dim)
        self.fc = nn.Linear(self.dim, self.num_classes)
        
        # 아웃풋의 위치 인코딩(1D 텍스트 데이터)
        self.dec_pos_encoder = PositionalEncoding(d_model=self.dim, max_len=self.max_output_length)
        self.y_mask = generate_square_subsequent_mask(self.max_output_length)
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim, nhead=tf_nhead, dim_feedforward=tf_fc_dim, dropout=tf_dropout),
            num_layers=tf_layers,
        )
        
        self.init_weights()  # This is empirically important
        
    def init_weights(self):
        """ 임의의 설정으로 가중치 초기화, 
            실험적(경험적)으로 중요한 역할을 한다.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        
        nn.init.kaiming_normal_(self.encoder_projection.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        
        if self.encoder_projection.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(  # pylint: disable=protected-access
                self.encoder_projection.weight.data
            )
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.encoder_projection.bias, -bound, bound)
            
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (Sx, B, E) logits
        """
        _B, C, _H, _W = x.shape # 이미지 데이터 차원
        
        # resnet은 3채널(rgb) 이미지 데이터가 들어오는 것을 예상한다.
        # iam 흑백 이미지를 사용하므로 채널 수를 3개로 늘려서 사용한다.
        if C == 1: 
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x) # (B, RESNET_DIM, _H // 32, _W // 32),   (B, 512, 18, 20) in the case of IAMParagraphs
        x = self.encoder_projection(x) # (B, E(self.dim), _H // 32, _W // 32),   (B, 256, 18, 20) in the case of IAMParagraphs
        
        # x = x * math.sqrt(self.dim)  # (B, E, _H // 32, _W // 32)  # This prevented any learning
        x = self.enc_pos_encoder(x)  # (B, E, Ho, Wo);     Ho = _H // 32, Wo = _W // 32
        x = torch.flatten(x, start_dim=2)  # (B, E, Ho * Wo)
        x = x.permute(2, 0, 1)  # (Sx, B, E);    Sx = Ho * Wo
        return x
    
    def decode(self, x, y):
        """
        Parameters
        ----------
        x
            torch.Tensor
            (Sx, B, E) logits
        y
            (B, Sy) with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (Sy, B, C) logits
        """
        y_padding_mask = y == self.padding_token
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E) (??) 
        # math.sqrt(self.dim) 백터의 크기로 나눠줌으로써 크기가 너무 커지는 것을 방지 => softmax함수의 gradient vanishing 문제를 방지한다.
        y = self.dec_pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(x)
        output = self.transformer_decoder(
            tgt=y, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=y_padding_mask
        )  # (Sy, B, E) (??) memory? 입력 인자 의미 알아두기
        
        output = self.fc(output)  # (Sy, B, C)
        return output
        
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image
        y
            (B, Sy) with elements in [0, C-1] where C is num_classes

        Returns
        -------
        """
        
        x = self.encode(x)  # (Sx, B, E)
        output = self.decode(x, y)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        
        B = x.shape[0]
        S = self.max_output_length
        x = self.encode(x)  # (Sx, B, E)
        
        output_tokens = (torch.ones((B, S)) * self.padding_token).type_as(x).long()  # (B, S)
        output_tokens[:, 0] = self.start_token  # Set start token
        
        for Sy in range(1, S):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(x, y)  # (Sy, B, C) C: class(문자) 별 확률 
            output = torch.argmax(output, dim=-1)  # (Sy, B) 확률이 제일 큰 인덱스 추출
            output_tokens[:, Sy : Sy + 1] = output[-1:]  # Set the last output token
            
            # 배치 내의 모든 토큰이 end이거나 padding이면 루프 탈출
            if ((output_tokens[:, Sy] == self.end_token) | (output_tokens[:, Sy] == self.padding_token)).all():
                break
            
        # end 토큰 이후 모든 자리에 padding 토큰 삽입
        for Sy in range(1, S):
            ind = (output_tokens[:, Sy - 1] == self.end_token) | (output_tokens[:, Sy - 1] == self.padding_token)
            output_tokens[ind, Sy] = self.padding_token
        
        return output_tokens  # (B, Sy)
        
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tf_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_fc_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_dropout", type=float, default=TF_DROPOUT)
        parser.add_argument("--tf_layers", type=int, default=TF_LAYERS)
        parser.add_argument("--tf_nhead", type=int, default=TF_NHEAD)
        return parser
        