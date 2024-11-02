# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig
import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# 모델 관련 인자를 저장하기 위한 데이터 클래스
@dataclass
class ModelArguments:
    # 사용할 모델의 이름 또는 경로
    # 기본값으로 "Qwen/Qwen-7B" 설정
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


# 데이터 관련 인자를 저장하기 위한 데이터 클래스
@dataclass
class DataArguments:
    # 훈련 데이터의 경로
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # 평가 데이터의 경로
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    # 지연 전처리 여부를 결정하는 플래그
    lazy_preprocess: bool = False


# 훈련 관련 인자를 저장하기 위한 데이터 클래스
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # 캐시 디렉토리 경로
    cache_dir: Optional[str] = field(default=None)
    # 최적화 알고리즘 선택 (기본값: adamw_torch)
    optim: str = field(default="adamw_torch")
    # 최대 시퀀스 길이 설정: 해당 길이를 초과하는 시퀀스는 오른쪽 패딩 또는 잘림 처리됨
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # LoRA (Low-Rank Adaptation) 사용 여부
    use_lora: bool = False
    # Vision Transformer (ViT) 부분을 고정할지 여부
    fix_vit: bool = True


# LoRA (Low-Rank Adaptation) 관련 인자를 저장하기 위한 데이터 클래스
# 이 클래스는 LoRA 기법을 적용할 때 필요한 다양한 설정값들을 관리합니다.
@dataclass
class LoraArguments:
    # LoRA의 랭크 (rank) 설정
    # LoRA 적용 시 사용되는 저차원 행렬의 랭크를 결정하며 높은 값은 더 많은 표현력 제공 및 계산 비용이 증가.
    # 기본값 64는 대부분의 경우에 좋은 성능과 효율성의 균형을 제공합니다.
    lora_r: int = 64

    # LoRA의 알파 값 설정
    # 이 값은 LoRA 업데이트의 크기를 조절합니다. lora_alpha / lora_r 비율이 실제 스케일링 팩터로 사용됩니다.
    # 기본값 16은 lora_r과 함께 1/4의 스케일링 팩터를 만듭니다.
    lora_alpha: int = 16

    # LoRA의 드롭아웃 비율 설정
    # 이는 LoRA 레이어에 적용되는 드롭아웃의 확률입니다. 0.05 (5%)의 값은 약간의 정규화 효과로 과적합을 줄입니다.
    lora_dropout: float = 0.05

    # LoRA를 적용할 타겟 모듈 리스트
    # 이 리스트는 LoRA가 적용될 모델의 특정 부분(레이어)을 지정합니다.
    # "c_attn", "attn.c_proj"는 주로 attention 메커니즘 관련 레이어입니다.
    # "w1", "w2"는 주로 feed-forward 네트워크의 가중치 행렬을 나타냅니다.
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # LoRA 가중치 파일 경로
    # 사전 훈련된 LoRA 가중치를 로드할 파일 경로. 빈 값인 경우 사전 훈련된 LoRA 가중치 미사용.
    lora_weight_path: str = ""

    # LoRA 편향(bias) 설정 방식
    # "none": 편향을 LoRA 적용에서 제외
    # "all": 모든 편향에 LoRA 적용
    # "lora_only": LoRA가 적용된 레이어의 편향에만 LoRA 적용
    # 기본값 "none"은 편향을 그대로 두어 모델의 안정성을 유지합니다.
    lora_bias: str = "none"

    # QLoRA(Quantized LoRA) 사용 여부
    # QLoRA는 모델의 크기를 줄이고 추론 속도를 높이지만 정확도 손실이 있을 수 있습니다.
    q_lora: bool = False


# DeepSpeed ZeRO-3 최적화와 관련된 파라미터 처리 함수
def maybe_zero_3(param):
    # 파라미터가 ZeRO-3에 의해 분할되었는지 확인
    if hasattr(param, "ds_id"):
        # 파라미터가 현재 사용 불가능한 상태인지 확인
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        # 파라미터를 수집하여 CPU로 이동
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        # ZeRO-3가 적용되지 않은 경우 단순히 CPU로 이동
        param = param.detach().cpu().clone()
    return param


# PEFT (Parameter-Efficient Fine-Tuning) 모델의 상태를 추출하는 함수
# 이 함수는 LoRA (Low-Rank Adaptation) 파라미터와 관련 편향을 효율적으로 추출함
# ZeRO-3 최적화가 적용된 경우에도 올바르게 처리 가능
def get_peft_state_maybe_zero_3(named_params, bias):
    # 편향 처리 방식에 따라 다른 로직 적용
    if bias == "none":
        # LoRA 관련 파라미터만 추출
        # 키에 "lora_"가 포함된 모든 파라미터를 선택
        # 이는 LoRA 레이어의 가중치만을 저장하고자 할 때 사용
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        # LoRA 관련 파라미터와 모든 편향 추출
        # 키에 "lora_" 또는 "bias"가 포함된 모든 파라미터를 선택
        # 이는 LoRA 레이어와 모든 편향을 함께 저장하고자 할 때 사용
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # LoRA 관련 파라미터와 LoRA에 해당하는 편향만 추출
        # 이는 LoRA 레이어와 그에 직접 관련된 편향만을 선택적으로 저장하고자 할 때 사용
        to_return = {}  # LoRA 파라미터와 관련 편향을 저장할 딕셔너리
        maybe_lora_bias = {}  # 모든 편향을 임시 저장할 딕셔너리
        lora_bias_names = set()  # LoRA 관련 편향 이름을 저장할 집합
        for k, t in named_params:
            if "lora_" in k:
                # LoRA 파라미터 저장
                to_return[k] = t
                # 해당 LoRA 파라미터에 대응하는 편향 이름 생성 및 저장
                # 예: "lora_layer1" -> "layer1_bias"
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                # 모든 편향 파라미터를 임시 저장
                maybe_lora_bias[k] = t
        # LoRA와 관련된 편향만 선택하여 최종 결과에 추가
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        # 지원하지 않는 bias 옵션인 경우 예외 발생
        raise NotImplementedError

    # ZeRO-3 최적화가 적용된 경우 파라미터 처리
    # maybe_zero_3 함수를 사용하여 각 파라미터를 처리
    # ZeRO-3는 메모리 효율을 위해 파라미터를 여러 GPU에 분산 저장함
    # 이 과정은 분산된 파라미터를 올바르게 수집하여 하나로 모으는 역할을 함
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return  # 처리된 LoRA 파라미터와 편향을 반환

local_rank = None

# 분산 학습 환경에서 첫 번째 GPU(rank 0)에서만 출력을 수행하는 함수
# - if local_rank == 0: 각 GPU의 고유 번호 (0, 1, 2, ...) 즉, 첫 번째 GPU에서만 조건이 참이 됨
# - *args: 여러 인자를 받아 그대로 print에 전달
# 목적: 첫번째 GPU 기준으로만 출력하여 여러 GPU에서의 중복 출력을 방지하고 로그를 깔끔하게 유지
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# Hugging Face Trainer를 위한 안전한 모델 저장 함수
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Trainer 객체에서 모델 상태를 수집하고 디스크에 저장합니다."""
    # DeepSpeed ZeRO-3 모드: 대형 모델을 여러 GPU에 나누어 저장하는 기술
    # 풀 파인 튜닝이나 LoRA 사용 후 모델 저장 시 이 나눠진 부분들을 다시 모아야 함
    if deepspeed.is_deepspeed_zero3_enabled():
        # 분산된 모델 파라미터를 하나로 모아 16비트 정밀도로 저장
        # ZeRO-3 + LoRA 또는 ZeRO-3 + 풀 파인튜닝 모두 해당
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        # (ZeRO-3 비활성화 상태)
        if trainer.args.use_lora:
            # LoRA: 모델의 일부만 미세조정하는 기법
            # LoRA 관련 파라미터만 추출하여 저장 
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            # 일반적인 경우: DeepSpeed ZeRO-3나 LoRA를 사용하지 않을 때
            # 모델의 전체 파라미터를 저장
            state_dict = trainer.model.state_dict()
    
    # 분산 학습 환경에서 모델 저장
    # - 여러 GPU에서 학습했어도 저장은 한 번만 수행
    # - local_rank 0: 첫 번째 GPU를 의미
    # - state_dict: 이미 전체 모델 정보가 통합되어 있음
    # - 첫 번째 GPU에서만 저장하여 중복 방지 및 일관성 유지
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}  # 역할별 토큰 구분

    im_start = tokenizer.encode("<|im_start|>")[0]  # 대화 시작 토큰
    im_end = tokenizer.encode("<|im_end|>")[0]  # 대화 끝 토큰
    nl_tokens = tokenizer('\n').input_ids  # 줄바꿈 토큰

    # 시스템, 유저, 어시스턴트 역할에 대한 토큰
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_ids, targets = [], []  # 입력과 타겟 저장 리스트

    for i, source in enumerate(sources):
        # 첫 메시지가 user가 아니면 건너뜀
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []

        # 시스템 메시지 추가
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens

        # 대화 데이터 처리
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]  # 역할 확인 (user 또는 assistant)
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id

            # 타겟 설정: user는 IGNORE_TOKEN_ID로 처리, assistant는 예측 대상
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError

            target += _target

        # 길이 맞추기 (패딩 추가)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids.append(input_id[:max_len])  # 최대 길이만큼 자르기
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int)  # 입력 시퀀스
    targets = torch.tensor(targets, dtype=torch.int)  # 타겟 시퀀스

    # 예시) 주어진 source:
    # sources = [
    #     [
    #         {"from": "user", "value": "Hello, how are you?"},
    #         {"from": "assistant", "value": "I'm doing well, thank you!"}
    #     ]
    # ]

    # 디코딩 예상 결과:
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Hello, how are you?<|im_end|>
    # <|im_start|>assistant
    # I'm doing well, thank you!<|im_end|>

    # 예상 input_ids:
    # [im_start, 'system' 토큰들, 'You are a helpful assistant.' 토큰들, im_end, nl_token,
    # im_start, 'user' 토큰들, 'Hello, how are you?' 토큰들, im_end, nl_token,
    # im_start, 'assistant' 토큰들, 'I'm doing well, thank you!' 토큰들, im_end, 패딩...]

    # 예상 targets:
    # targets에서는 시스템과 user의 대화는 전부 IGNORE하는게 목적. assistant의 발화만 예측.
    # [im_start, IGNORE_TOKEN_ID들, im_end, nl_token,
    # im_start, IGNORE_TOKEN_ID들, im_end, nl_token,
    # im_start, IGNORE_TOKEN_ID들, 'I'm doing well, thank you!' 토큰들, im_end, 패딩...]

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # 패딩된 부분을 무시하도록 설정
    )

# SupervisedDataset
# - 전처리 시점: 학습 전에 모든 데이터를 한 번에 전처리 (데이터셋 생성 시 전처리 완료)
# - 장점: 학습 중 빠르게 데이터를 불러올 수 있음
# - 단점: 메모리 사용량이 많아지고, 초기 전처리 시간이 길어질 수 있음

# LazySupervisedDataset
# - 전처리 시점: 학습 중 필요한 데이터에 접근할 때마다 전처리
# - 장점: 메모리 효율적이고, 초기 로딩 속도가 빠름 <= 초기 로딩 속도가 빠른 것이 이점.
# - 단점: 학습 중 데이터에 처음 접근할 때 전처리 시간이 그때 발생함.
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


# 지도 학습을 위한 데이터 모듈 생성 함수
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """지도 학습을 위한 데이터셋과 콜레이터를 생성합니다."""
    # 지연 처리 여부에 따라 데이터셋 클래스 선택
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # 훈련 데이터 로드 및 데이터셋 생성
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    # 평가 데이터가 제공된 경우 평가 데이터셋도 생성
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# 메인 훈련 함수
def train():
    # global 키워드를 사용하여 local_rank 변수를 전역 범위에서 사용할 수 있게 합니다.
    # 이는 분산 학습 환경에서 현재 프로세스의 랭크를 전역적으로 관리하기 위함입니다.
    global local_rank
    
    # HfArgumentParser를 사용하여 명령줄 인자를 파싱합니다.
    # 이 파서는 복잡한 설정을 쉽게 관리할 수 있게하기 위함입니다.
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    # parse_args_into_dataclasses 메서드를 사용하여 인자를 파싱하고 각각의 데이터클래스 인스턴스로 변환합니다.
    # 타입 안정성과 코드의 구조화를 개선할 수 있습니다.
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # DeepSpeed와 Q-LoRA(Quantized LoRA) 설정을 확인하고 처리합니다.
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        # DeepSpeed와 Q-LoRA를 동시에 사용하는 경우, 분산 학습 타입을 DeepSpeed으로 명시적으로 설정합니다.
        # 이는 두 기술의 호환성을 보장하고 최적의 성능을 얻기 위함입니다.
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # 계산에 사용할 데이터 타입을 결정. fp16(반정밀도), bf16(bfloat16), fp32(단정밀도) 중 택1
    # 훈련 속도와 메모리 사용량, 그리고 수치 안정성에 영향을 줍니다.
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # local_rank: 각 GPU 프로세스의 고유 식별자. 
    # 멀티 GPU: 각 프로세스는 0부터 (GPU 개수 - 1)까지의 값을 가짐.
    # 예: 4 GPU 시스템에서 각 프로세스는 0, 1, 2, 3 중 하나의 값을 가짐.
    # 단일 GPU: 보통 -1 또는 0.    
    # 이 값은 프로세스마다 다르며, 각 프로세스는 자신의 local_rank만 알고 있음.
    local_rank = training_args.local_rank

    # device_map: 모델 층을 GPU에 배치하는 방법을 지정. None은 기본 배치 사용.
    device_map = None

    # world_size: 전체 분산 학습에 참여하는 프로세스 총 개수.
    # 단일 GPU: 1
    # 멀티 GPU: GPU 개수와 동일 (예: 4 GPU 사용 시 모든 프로세스에서 world_size는 4)
    # int()로 문자열을 정수로 변환, 환경변수 없으면 기본값 1 사용.
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ddp: DistributedDataParallel 사용 여부. world_size > 1이면 True.
    # 단일 GPU: False
    # 멀티 GPU: True (예: 4 GPU 사용 시 모든 프로세스에서 ddp는 True)
    ddp = world_size != 1
    if lora_args.q_lora:
        # Q-LoRA 사용 시 device_map 설정:
        # ddp가 True(멀티 GPU)면 각 프로세스가 하나의 전체 GPU를 사용.
        # LOCAL_RANK 환경변수로 GPU 지정, 없으면 0 사용.
        # ddp가 False(단일 GPU)면 None 유지.
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        # FSDP 또는 DeepSpeed ZeRO-3 활성화 확인.
        # 이들은 Q-LoRA와 호환성 문제가 있어 경고 발생.
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # 모델 설정을 로드합니다.
    # AutoConfig를 사용하여 사전 훈련된 모델의 설정을 자동으로 로드합니다.
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    # 캐시 사용을 비활성화합니다. 이는 훈련 중 메모리 사용을 최적화하기 위함입니다.
    config.use_cache = False

    # 모델을 로드합니다.
    # AutoModelForCausalLM을 사용하여 인과적 언어 모델(GPT 계열)을 자동으로 로드합니다.
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        # Q-LoRA를 사용하는 경우 4비트 양자화를 적용합니다.
        # 이는 모델의 메모리 사용량을 크게 줄이면서 성능은 유지할 수 있게 합니다.
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    # LoRA를 사용하지 않는 경우, Vision Transformer(ViT) 부분을 고정합니다.
    # 이는 멀티모달 모델에서 비전 부분의 가중치를 동결하여 훈련 효율성을 높이기 위함입니다.
    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
            model.transformer.visual.requires_grad_(False)
            # 하지만 attention pooling 층은 훈련 가능하게 둡니다.
            if hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)

    # 토크나이저를 로드합니다.
    # AutoTokenizer를 사용하여 모델에 맞는 토크나이저를 자동으로 로드합니다.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # 패딩 토큰 ID를 EOD(End of Document) 토큰 ID로 설정합니다. 모델이 문서의 끝을 인식하도록 합니다.
    # tokenizer.pad_token_id = tokenizer.eod_id

    # LoRA 설정 및 적용
    # LoRA는 대규모 언어 모델의 효율적인 미세조정을 가능하게 하는 기술입니다.
    if training_args.use_lora:
        # Q-LoRA를 사용하거나 채팅 모델인 경우 특별히 저장할 모듈을 지정하지 않습니다.
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            # 그 외의 경우, 토큰 임베딩과 언어 모델 헤드를 저장할 모듈로 지정합니다.
            modules_to_save = ["wte", "lm_head"]
        
        # LoRA 설정을 구성합니다.
        lora_config = LoraConfig(
            r=lora_args.lora_r,  # LoRA의 랭크
            lora_alpha=lora_args.lora_alpha,  # LoRA의 알파 값
            target_modules=lora_args.lora_target_modules,  # LoRA를 적용할 대상 모듈
            lora_dropout=lora_args.lora_dropout,  # LoRA 드롭아웃 비율
            bias=lora_args.lora_bias,  # 편향 학습 방식
            task_type="CAUSAL_LM",  # 작업 유형 (인과적 언어 모델링)
            modules_to_save=modules_to_save  # 추가로 저장할 모듈
        )
        
        # Q-LoRA를 사용하는 경우, 모델을 k-비트 훈련에 맞게 준비합니다.
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        # PEFT 모델로 변환합니다. 이는 LoRA 레이어를 모델에 추가합니다.
        model = get_peft_model(model, lora_config)

        # 그래디언트 체크포인팅을 사용하는 경우, 입력에 대한 그래디언트 계산을 활성화합니다.
        # 이는 메모리 효율성을 높이지만 계산 속도가 약간 느려질 수 있습니다.
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # 지도 학습을 위한 데이터셋을 준비.
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Trainer를 초기화하고 훈련을 시작. Trainer 클래스는 훈련 루프, 평가, 로깅 등을 자동으로 처리.
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # 실제 훈련을 수행.
    trainer.train()
    # 훈련 상태를 저장. 이는 중단된 훈련을 이어서 할 수 있게 해줍니다.
    trainer.save_state()

    # 이 함수는 DeepSpeed ZeRO-3와 같은 고급 최적화 기법을 고려하여 모델을 안전하게 저장합니다.
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)

# 스크립트가 직접 실행될 때 train 함수를 호출합니다.
if __name__ == "__main__":
    train()