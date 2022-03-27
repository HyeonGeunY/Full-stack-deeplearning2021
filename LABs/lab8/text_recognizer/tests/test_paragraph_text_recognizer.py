"""Test for paragraph_text_recognizer module.
지정한 이미지 입력에 출력이 문제가 없는지 테스트한다.

data_by_file_id.json파일에 아래와 같이 인풋 이미지 id, 실제 텍스트(ground_truth_text), test에서 예측했던 output(predicted_text), cer(cer)을 기록해둔다.

테스트 코드를 통해 모델의 동작이 일관되게 나오는 지 확인한다. (json 파일의 predicted_text코드와 모델 아웃풋 비교)

json파일의 ground truth와 모델 아웃풋을 비교하여 cer을 출력한다.

모델이 로드되는데 걸리는 시간과 예측에 걸리는 시간을 출력한다.
"""

import os
import json
from pathlib import Path
import time
import editdistance
from text_recognizer.paragraph_text_recognizer_source import ParagraphTextRecognizer_source


# gpu 사용 안함.
os.environ["CUDA_VISIBLE_DEVICES"] = ""



# 현재 디렉토리의 절대 경로 반환
_FILE_DIRNAME = Path(__file__).parents[0].resolve()
_SUPPORT_DIRNAME = _FILE_DIRNAME / "support" / "paragraphs"

# CircleCI의 timeout을 방지하기 위해 샘플 수를 제한
_NUM_MAX_SAMPLES = 2 if os.environ.get("CIRCLECI", False) else 100


def test_paragraph_text_recognizer():
    """Test ParagraphTextRecognizer."""
    support_filenames = list(_SUPPORT_DIRNAME.glob("*.png"))
    with open(_SUPPORT_DIRNAME / "data_by_file_id.json", "r") as f:
        support_data_by_file_id = json.load(f)

    start_time = time.time()
    text_recognizer = ParagraphTextRecognizer_source()
    end_time = time.time()
    # initialize 하는데 걸리는 시간
    print(f"Time taken to initialize ParagraphTextRecognizer: {round(end_time - start_time, 2)}s")

    for i, support_filename in enumerate(support_filenames):
        # 제한한 숫자보다 많은 샘플이 들어올 경우 탈출
        if i >= _NUM_MAX_SAMPLES:
            break
        expected_text = support_data_by_file_id[support_filename.stem]["predicted_text"]
        start_time = time.time()
        predicted_text = _test_paragraph_text_recognizer(support_filename, expected_text, text_recognizer)
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)

        cer = _character_error_rate(support_data_by_file_id[support_filename.stem]["ground_truth_text"], predicted_text)
        print(f"Character error rate is {round(cer, 3)} for file {support_filename.name} (time taken: {time_taken}s)")


def _test_paragraph_text_recognizer(image_filename: Path, expected_text: str, text_recognizer: ParagraphTextRecognizer_source):
    """Test ParagraphTextRecognizer on 1 image."""
    predicted_text = text_recognizer.predict(image_filename)
    assert predicted_text == expected_text, f"predicted text does not match expected for {image_filename.name}"
    return predicted_text


def _character_error_rate(str_a: str, str_b: str) -> float:
    """Return character error rate."""
    return editdistance.eval(str_a, str_b) / max(len(str_a), len(str_b))
