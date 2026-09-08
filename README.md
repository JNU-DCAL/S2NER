# S2NER

Historical Chinese Named Entity Recognition baseline using **BIO tagging + CRF**.

* Entity types: `PER`, `LOC`, `BOOK`
* Backbones:

  * `ddokbaro/SillokBert`
  * `KoichiYasuoka/roberta-classical-chinese-base-char`
  * `SIKU-BERT/sikuroberta`

## 📁 Structure

```text
S2NER/
├── baseline/
│   ├── common.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── run_*.sh
├── tests/
│   ├── test_baseline.py
│   └── test_launchers.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/JNU-DCAL/S2NER.git
cd S2NER

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 기준입니다.

## 📦 Dataset

다음 세 파일을 하나의 폴더에 배치합니다.

```text
Sillok_train_final.jsonl
Sillok_dev_final.jsonl
SJW.jsonl
```

```bash
export DATA_DIR=/path/to/data
```

각 JSONL line은 하나의 paragraph이며 character-level BIO annotation을 포함합니다.

```json
{
  "id": "example-0",
  "text": "李京",
  "bio": [
    {"char": "李", "tag": "B-PER"},
    {"char": "京", "tag": "B-LOC"}
  ]
}
```

## 🚀 Training

4-GPU 기준 실행:

```bash
export DATA_DIR=/path/to/data
export GPUS=0,1,2,3
export OUTPUT_ROOT="$PWD/outputs"

bash baseline/run_experiments.sh all
```

명령만 확인:

```bash
DRY_RUN=1 bash baseline/run_experiments.sh all
```

개별 모델 실행:

```bash
bash baseline/run_sillokbert.sh
bash baseline/run_classical_chinese.sh
bash baseline/run_siku.sh
```

기본 설정:

| Setting        |       Value |
| -------------- | ----------: |
| Epochs         |           4 |
| Learning rate  |      `6e-5` |
| Train batch    | `128 / GPU` |
| Precision      |        BF16 |
| Max length     |         510 |
| Window overlap |         256 |

## 📊 Evaluation

저장된 checkpoint 평가:

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/evaluate.py \
  --model sillokbert \
  --checkpoint outputs/sillokbert_crf/best_model \
  --train_jsonl "$DATA_DIR/Sillok_train_final.jsonl" \
  --dev_jsonl "$DATA_DIR/Sillok_dev_final.jsonl" \
  --test_jsonl "$DATA_DIR/SJW.jsonl" \
  --output_dir outputs/sillokbert_reeval
```

PER / LOC / BOOK의 precision, recall, F1과 micro/macro F1, seen/unseen 성능을 출력합니다.

## 🧪 Tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```

Synthetic data와 작은 local BERT를 사용하므로 실제 dataset이나 GPU 없이 실행할 수 있습니다.
