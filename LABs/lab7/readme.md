# Lab 7: Paragraph Recognition

*Thanks to [Saurabh Bipin Chandra](https://www.linkedin.com/in/srbhchandra/) for extensive work on this lab!*

IAM 데이터셋을 사용하여 IAM_lines, IAM_pragraphs, IAM_synthetic_paragraphs를 만들어서 훈련에 사용한다.
학습에 사용할 훈련 데이터 셋을 늘리기 위해 IAM_synthetic_paragraphs를 trainset에 추가한다. (text_recongnizer/data/iam_original_and_synthetic_paragraphs.py)

이미지에서 특징 벡터를 추출하기 위해 resnet과 kernel_size=1의 conv layer를 사용한다.
2D positional 인코딩을 이용하여 문장을 학습할 때와 같이 문단 이미지 자체에서 위치정보를 처리해줄 수 있다.

실행 코드 예시 아래와 같다.

```sh
python training/run_experiment.py --wandb --gpus=-1 --data_class=IAMOriginalAndSyntheticParagraphs --model_class=ResnetTransformer --loss=transformer --batch_size=16 --check_val_every_n_epoch=10 --terminate_on_nan=1 --num_workers=24 --accelerator=ddp --lr=0.0001 --accumulate_grad_batches=4
```


아래 명령어로 sweep을 통해 확습 시킨다.
```sh
wandb sweep training/sweeps/IAMparagraphs_resnet_transformer.yml
```

```sh
wandb sweep training/sweeps/emnist_lines2_line_cnn_transformer.yml
```


# Error
## AttributeError: 'IAMParagraphs' object has no attribute 'data_test'
초기 단계에 준비하지 않은 test 데이터를 요구하여 생기는 문제로 추정

iam_original_and_synthetic_paragraphs.py 43번째 line의
```sh
self.data_test = self.iam_paragraphs.data_test
```
코드에 아래와 같은 조건문을 추가하여 해결 
```sh
if stage == 'test' or stage is None:
    self.data_test = self.iam_paragraphs.data_test
```

    
##  RuntimeError: The expanded size of the tensor (1) must match the existing size (8) at non-singleton dimension 1.  Target sizes: [8, 1].  Tensor sizes: [8]    
\
텐서의 차원이 맞지 않아서 발생
```sh
output_tokens[:, Sy : Sy + 1] = output[-1:]
```
를 아래와 같이 수정하여 해결

```sh
output_tokens[:, Sy : Sy + 1] = output[-1:].unsqueeze(-1)
```







