if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from typing import Sequence, Union
import argparse
import json

from PIL import Image
import torch

from text_recognizer.data import IAMParagraphs
from text_recognizer.data.iam_paragraphs import resize_image, IMAGE_SCALE_FACTOR, get_transform
from text_recognizer.lit_models import TransformerLitModel
from text_recognizer.models import ResnetTransformer
import text_recognizer.util as util


CONFIG_AND_WEIGHTS_DIRNAME = (
    Path(__file__).resolve().parent / "artifacts" / "paragraph_text_recognizer"
)


class ParagraphTextRecognizer:
    """Class to recognize paragraph text in an image."""

    def __init__(self):
        data = IAMParagraphs()
        self.mapping = data.mapping  # 기존 mapping에 "\n"가 추가되어 있는 mapping

        # 제품은 추론 단계만 진행하므로 label -> str 정보만 있으면 된다.
        # ignore_tokens에 해당하는 인덱스 정보만 따로 선언한다.
        inv_mapping = data.inverse_mapping
        self.ignore_tokens = [
            inv_mapping["<S>"],
            inv_mapping["<B>"],
            inv_mapping["<E>"],
            inv_mapping["<P>"],
        ]
        self.transform = get_transform(image_shape=data.dims[1:], augment=False)

        with open(CONFIG_AND_WEIGHTS_DIRNAME / "config.json", "r") as file:
            config = json.load(file)
        args = argparse.Namespace(**config)

        # 모델 형태를 만들고 가중치를 로드해 업데이트 하는 방식으로 모델을 불러온다.
        model = ResnetTransformer(data_config=data.config(), args=args)
        self.lit_model = TransformerLitModel.load_from_checkpoint(
            checkpoint_path=CONFIG_AND_WEIGHTS_DIRNAME / "model.pt", args=args, model=model
        )

        # 평가모드로 전환
        self.lit_model.eval()

        # 평가모드로 전환한 모델을 배포를 위한 torch script로 전환한다.
        self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image, grayscale=True)

        image_pil = resize_image(image_pil, IMAGE_SCALE_FACTOR)
        image_tensor = self.transform(image_pil)

        y_pred = self.scripted_model(image_tensor.unsqueeze(axis=0))[0]
        pred_str = convert_y_label_to_string(
            y=y_pred, mapping=self.mapping, ignore_tokens=self.ignore_tokens
        )

        return pred_str


def convert_y_label_to_string(
    y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]
) -> str:
    """
    추론된 레이블 값을 텍스트로 변환한다.
    """
    return "".join([mapping[i] for i in y if i not in ignore_tokens])


def main():
    """
    Example runs:
    ```
    python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
    python text_recognizer/paragraph_text_recognizer.py https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
    """
    parser = argparse.ArgumentParser(description="Recognize handwritten text in an image file.")
    # filename을 parser의 위치 인자로 사용
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    text_recognizer = ParagraphTextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
