"""
주어진 데이터 셋에서 훈련된 best모델을 artifacts 디렉토리에 저장

Run:
python training/save_best_model.py --entity=fsdl-user \
                                   --project=fsdl-text-recognizer-2021-labs \
                                   --trained_data_class=IAMLines

entity와 project 이름은 wandb 웹에 접속한 후 run의 overview 페이지에서 "run path"를 보면 확인할 수 있다.
"Run path" 의 포멧은 다음과 같다. <entity>/<project>/<run_id>".
"""
import argparse
import sys
import shutil
import json
from pathlib import Path
import tempfile
from typing import Optional, Union
import wandb


FILE_NAME = Path(__file__).resolve()
ARTIFACTS_BASE_DIRNAME = FILE_NAME.parents[1] / "text_recognizer" / "artifacts"
TRAINING_LOGS_DIRNAME = FILE_NAME.parent / "logs"


def save_best_model():
    """Find and save the best model trained on a given dataset to artifacts directory."""
    parser = _setup_parser()
    args = parser.parse_args()
    
    # mode에 따라 내림차순으로 정렬할지 오른차순으로 정렬할지 정한다.    
    # data_class를 사용한 run들을 matric 값으로 정렬한 후 맨 앞 run 정보를 가져오는 방식으로 최적의 run을 가져온다.
    if args.mode == "min":
        default_metric_value = sys.maxsize
        sort_reverse = False
    else:
        default_metric_value = 0
        sort_reverse = True
        
    api = wandb.Api()
    # filter에 따라 runs 불러오기
    runs = api.runs(f"{args.entity}/{args.project}", filters={"config.data_class": args.trained_data_class})
    # 훈련을 진행한 runs 중에서 key(metric)에 해당하는 값을 기준으로 정렬한다.
    sorted_runs = sorted(
        runs,
        key=lambda run: _get_summary_value(wandb_run=run, key=args.metric, default=default_metric_value),
        reverse=sort_reverse,
    )

    best_run = sorted_runs[0]
    summary = best_run.summary
    # name, id
    print(f"Best run ({best_run.name}, {best_run.id}) picked from {len(runs)} runs with the following metrics:")
    print(f" - val_loss: {summary['val_loss']}, val_cer: {summary['val_cer']}, test_cer: {summary['test_cer']}")

    artifacts_dirname = _get_artifacts_dirname(args.trained_data_class)
    with open(artifacts_dirname / "config.json", "w") as file:
        json.dump(best_run.config, file, indent=4)
        
    with open(artifacts_dirname / "run_command.txt", "w") as file:
        file.write(_get_run_command(best_run))
        
    _save_model_weights(wandb_run=best_run, project=args.project, output_dirname=artifacts_dirname)
    
    
def _get_artifacts_dirname(trained_data_class: str) -> Path:
    """Return artifacts dirname."""
    for keyword in ["line", "paragraph"]:
        if keyword in trained_data_class.lower():
            artifacts_dirname = ARTIFACTS_BASE_DIRNAME / f"{keyword}_text_recognizer"
            artifacts_dirname.mkdir(parents=True, exist_ok=True)
            break
    return artifacts_dirname


def _save_model_weights(wandb_run: wandb.apis.public.Run, project: str, output_dirname: Path):
    """Save checkpointed model weights in output_dirname.
       로컹에 파일이 있다면 로컬 파일을 output_dirname에 복사
       없다면 wandb 서버에서 파일을 다운로드하여 output_dirname에 복사한다.
    """
    weights_filename = _copy_local_model_checkpoint(run_id=wandb_run.id, project=project, output_dirname=output_dirname)
    if weights_filename is None:
        weights_filename = _download_model_checkpoint(wandb_run, output_dirname)
        assert weights_filename is not None, "Model checkpoint not found"


def _copy_local_model_checkpoint(run_id: str, project: str, output_dirname: Path) -> Optional[Path]:
    """Copy model checkpoint file on system to output_dirname."""
    checkpoint_filenames = list((TRAINING_LOGS_DIRNAME / project / run_id).glob("**/*.ckpt")) # logs/project_name/best_run_id 에 있는 checkpoints/~~.ckpt path을 가져옴
    if not checkpoint_filenames:
        return None
    shutil.copyfile(src=checkpoint_filenames[0], dst=output_dirname / "model.pt") # src[0](파일)을 dst에 저장 (model.pt)
    print(f"Model checkpoint found on system at {checkpoint_filenames[0]}")
    return checkpoint_filenames[0]


def _download_model_checkpoint(wandb_run: wandb.apis.public.Run, output_dirname: Path) -> Optional[Path]:
    """
       wandb 서버로부터 checkpoint 파일을 output_dirname에 다운로드
    """
    checkpoint_wandb_files = [file for file in wandb_run.files() if file.name.endswith(".ckpt")]
    if not checkpoint_wandb_files:
        return None

    wandb_file = checkpoint_wandb_files[0]
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file.download(root=tmp_dirname, replace=True)
        checkpoint_filename = f"{tmp_dirname}/{wandb_file.name}"
        shutil.copyfile(src=checkpoint_filename, dst=output_dirname / "model.pt")
        print("Model checkpoint downloaded from wandb")
    return output_dirname / "model.pt"


def _get_run_command(wandb_run: wandb.apis.public.Run) -> str:
    """Return python run command for input wandb_run."""
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file = wandb_run.file("wandb-metadata.json")
        with wandb_file.download(root=tmp_dirname, replace=True) as file:
            metadata = json.load(file)

    return f"python {metadata['program']} " + " ".join(metadata["args"])


def _get_summary_value(wandb_run: wandb.apis.public.Run, key: str, default: int) -> Union[int, float]:
    """
       key에 해당하는 수치 값(summary[key])을 반환한다. 값이 타당한 자료형이 아니면 디폴트 값을 반환한다.
    """
    value = wandb_run.summary.get(key, default)
    if not isinstance(value, (int, float)):
        value = default
    return value

    
def _setup_parser() -> argparse.ArgumentParser:
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments.
    
        entity: userid
        project: 프로젝트 이름
        trained_data_class: 훈련에 사용한 데이터 클래스
        metric: 최적 모델을 정할 기준 metric
        mode: metric의 최적을 정하는 기준
        
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", type=str, default="hgyoon0928")
    parser.add_argument("--project", type=str, default="Full-stack-deeplearning")
    parser.add_argument("--trained_data_class", type=str, default="IAMLines")
    parser.add_argument("--metric", type=str, default="val_loss")
    parser.add_argument("--mode", type=str, default="min")
    return parser


if __name__ == "__main__":
    save_best_model()
