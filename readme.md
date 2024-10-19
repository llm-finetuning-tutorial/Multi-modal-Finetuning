# 데이터셋 처리 및 모델 파인튜닝 프로젝트

이 프로젝트는 ScienceQA 데이터셋을 처리하고, 처리된 데이터를 사용하여 모델을 파인튜닝하는 과정을 포함합니다.


## 사용 방법

### 1. 데이터셋 처리

`preprocess.py` 스크립트는 ScienceQA 데이터셋을 로드하고 처리합니다. 다음과 같이 실행합니다:

```
python preprocess.py --output_path ./data --num_samples 200
```

- `--output_path`: 처리된 데이터를 저장할 디렉토리 경로
- `--num_samples`: 처리할 샘플의 수 (기본값: 200)

이 스크립트는 지정된 출력 경로에 `formatted_conversations.json` 파일을 생성합니다.

### 2. 모델 파인튜닝

`finetune_lora_single_gpu.sh` 스크립트는 처리된 데이터를 사용하여 모델을 파인튜닝합니다. 다음과 같이 실행합니다:

```
bash ./finetune/finetune_lora_single_gpu.sh
```

## 전체 프로세스 실행

데이터셋 처리와 모델 파인튜닝을 연속해서 실행하려면 다음 명령어를 사용합니다:

```
bash train.sh 
```

이 명령어는 먼저 데이터셋을 처리한 후, 처리된 데이터를 사용하여 모델을 파인튜닝합니다.
