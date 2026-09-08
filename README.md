# Historical Chinese NER: BIO + CRF baseline

실록 학습·검증 데이터와 승정원일기 평가 데이터를 사용하는 문자 단위
PER/LOC/BOOK 개체명 인식 코드입니다. 세 백본은 각각 독립적으로 학습합니다.
데이터와 모델 가중치는 이 코드 배포 폴더에 포함하지 않습니다.

## 파일 구성

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data_manifest.json
├── baseline/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── common.py
│   ├── run_experiments.sh
│   ├── run_sillokbert.sh
│   ├── run_classical_chinese.sh
│   └── run_siku.sh
├── docs/
│   ├── METRICS.md
│   └── VALIDATION.md
├── results/
│   └── reported_test_metrics.json
└── tests/
    ├── __init__.py
    ├── test_baseline.py
    └── test_launchers.py
```

`train.py`: 학습 → dev 최적 모델 선택 → test 평가.
`evaluate.py`: 저장된 체크포인트만 재평가하며 dev/test 공통 metric을 제공.
`inference.py`: CRF 추론 및 윈도우 병합 모듈(독립 CLI 아님).
`common.py`: BIO 로더·모델·실행 설정.
기존 `finetuning_ds.py`, `finetuning.py`, `ner_*.py`는 필요하지 않습니다.

## 설치

이 README가 있는 저장소 루트에서 실행합니다. Linux/Bash와 Python 3.10을
기준으로 검증했습니다. CUDA wheel은 GPU/드라이버에 맞게 설치하십시오.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python baseline/train.py --help
```

requirements는 로컬 검증 환경의 직접 의존성 고정값이며 전체 전이 의존성
lockfile은 아닙니다. 새 환경 설치 검증과 전체 GPU 재학습을 완료했다는 뜻도
아닙니다. 검증 환경의 PyTorch는 2.6.0+cu124입니다.
`pytorch-crf`와 fast tokenizer는 필수입니다.

## 데이터 연결

별도로 준비한 데이터 폴더에 다음 파일을 배치하고 `DATA_DIR`로 지정합니다.

| 파일 | 용도 |
|---|---|
| Sillok_train_final.jsonl | 학습 |
| Sillok_dev_final.jsonl | 검증·최적 체크포인트 선택 |
| SJW.jsonl | 최종 평가 |

문단 하나가 JSONL 한 줄입니다. 아래는 형식 설명용 합성 예시입니다.

```json
{"id":"example-0","text":"李京","bio":[{"char":"李","tag":"B-PER"},{"char":"京","tag":"B-LOC"}]}
```

`text`와 `bio`의 문자 수·순서가 같아야 합니다. 태그는 O와
B/I-PER, B/I-LOC, B/I-BOOK입니다. 추가 `meta`는 학습에 사용하지 않습니다.
데이터 자동 다운로드나 업로드는 하지 않습니다.
`data_manifest.json`은 기존 실험 파일의 해시 기록입니다. 별도 준비한
배포본이 변경되었다면 그 기록과 일치한다고 가정하지 말고 확인하십시오.

## 학습

```bash
export DATA_DIR=/path/to/bio-data
export GPUS=0,1,2,3
export OUTPUT_ROOT="$PWD/outputs"
# 필요하면 export PYTHON_BIN=/path/to/venv/bin/python

# GPU 작업 없이 명령 확인
DRY_RUN=1 bash baseline/run_experiments.sh all

# 세 모델 순차 실행
bash baseline/run_experiments.sh all
```

개별 실행은 `run_sillokbert.sh`, `run_classical_chinese.sh`, `run_siku.sh`를
사용합니다. all과 개별 실행을 같은 출력 경로에 중복 실행하지 마십시오.
실행 스크립트는 기록된 설정을 재구성한 것으로 과거 원본 셸 파일은 아닙니다.

| 별칭 | 백본 |
|---|---|
| sillokbert | ddokbaro/SillokBert |
| classical_chinese | KoichiYasuoka/roberta-classical-chinese-base-char |
| sikuroberta (siku) | SIKU-BERT/sikuroberta |

백본 revision은 `baseline/common.py`에 고정되어 있습니다.
공통 설정: 최대 LR 6e-5, linear scheduler, warmup 0.1, 4 epochs,
seed 42, train batch 128/GPU × 4 GPU = 512, accumulation 1, BF16,
eval batch 512/GPU, 윈도우 510 tokens와 중첩 256 tokens.
dev micro-F1이 가장 높은 체크포인트를 최종 평가에 사용합니다.
GPU 점유를 매 모델 시작 전에 확인하지만 자원 예약 기능은 아니므로,
공유 서버에서는 스케줄러 또는 사용자 간 조율이 필요합니다.

## 체크포인트 재평가

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/evaluate.py \
  --model sillokbert --checkpoint outputs/sillokbert_crf/best_model \
  --train_jsonl "$DATA_DIR/Sillok_train_final.jsonl" \
  --dev_jsonl "$DATA_DIR/Sillok_dev_final.jsonl" \
  --test_jsonl "$DATA_DIR/SJW.jsonl" \
  --output_dir outputs/sillokbert_reeval
```

`--checkpoint`는 필수이며 학습 당시 백본/토크나이저를 동일하게 지정합니다.
`--test_only` 없이 실행할 수 있고 학습은 수행하지 않습니다.
train/dev는 seen 집합 구성에 사용합니다.
`python -m baseline.train`과 `python -m baseline.evaluate`도 지원합니다.
출력 폴더가 비어 있지 않으면 중단하여 기존 결과를 보호합니다.

학습 출력: `run_config.json`, `checkpoint-*`, `best_model/`,
`best_checkpoint.json`, `dev_metrics.json`, `test_metrics.json`,
`test_predictions.jsonl`. 재평가는 test 점수와 예측을 저장합니다.
`results/reported_test_metrics.json`은 과거 실험의 결과이며 이 패키지로
새로 학습한 결과가 아닙니다. 과거 `*_support`는 기록 보존 목적으로만 남겼고,
새 실행은 support/subtype을 출력하지 않습니다.

## 테스트와 공개 정보

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

합성 데이터와 로컬 tiny BERT만 사용하며 실제 데이터·GPU·백본 다운로드는
필요하지 않습니다. [지표 정의](docs/METRICS.md),
[검증 기록](docs/VALIDATION.md)을 참고하십시오.

## 라이선스·인용·데이터 배포

코드 라이선스와 논문 서지정보는 아직 확정되지 않아 LICENSE와 CITATION.cff를
임의로 만들지 않았습니다. 공개 전 권리자·사용조건을 확인하고 해당 파일 및
별도로 준비한 데이터의 배포 URL을 추가하십시오.
