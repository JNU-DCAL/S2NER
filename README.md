# S2NER

**S2NER**은 조선왕조실록(Joseon Wangjo **S**illok)과 승정원일기(**S**eungjeongwon Ilgi)를 대상으로 한 고전 한문 문자 단위 개체명 인식(NER) baseline입니다.

* Entity types: `PER`, `LOC`, `BOOK`


## 📁 Structure

```text
S2NER/
├── baseline/
│   ├── common.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── run_*.sh
├── dictionaries/
│   ├── PER/
│   │   ├── PER_blocklist.csv
│   │   ├── PER_person.csv
│   │   ├── PER_title.csv
│   │   └── README_PER.md
│   ├── LOC/
│   │   ├── LOC_attachment.csv
│   │   ├── LOC_blocklist.csv
│   │   ├── LOC_building.csv
│   │   ├── LOC_exclusion.csv
│   │   ├── LOC_gazetteer.csv
│   │   ├── LOC_office.csv
│   │   └── README_LOC.md
│   └── BOOK/
│       ├── BOOK_component.csv
│       ├── BOOK_keep.csv
│       ├── BOOK_split.csv
│       └── README_BOOK.md
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

python -m venv S2NER
source S2NER/bin/activate
pip install -r requirements.txt
```

Python 3.10 기준입니다.

## 📦 Dataset

📎 [Download dataset from Google Drive](https://drive.google.com/drive/folders/1Q54ynRk38AWdIwCYnMLfEm-Qq657dqVR?usp=drive_link)

다음 파일이 포함되어 있습니다.

```text
Sillok_train.jsonl
Sillok_dev.jsonl
SJW_test.jsonl
```

각 JSONL line은 하나의 paragraph이며 character-level BIO annotation을 포함합니다.

## 📊 Experimental Results

조선왕조실록(Sillok)의 train/dev 데이터로 학습한 모델을 승정원일기(Seungjeongwon Ilgi) test 데이터에서 평가했습니다. 평가는 개체 경계와 유형이 모두 일치해야 정답으로 인정하는 exact-span F1을 사용합니다.

| Dataset       | Model |   PER F1 |   LOC F1 |  BOOK F1 | Macro F1 |
| ------------- | ----- | -------: | -------: | -------: | -------: |
| Orig.         | SB    |     97.2 |     65.9 |     67.3 |     76.8 |
| Orig.         | SIKU  |     97.1 |     65.6 |     65.4 |     76.0 |
| Orig.         | CCR   |     96.9 |     65.7 |     66.4 |     76.3 |
| Retag. (Ours) | SB    | **97.9** | **78.0** | **71.6** | **82.4** |
| Retag. (Ours) | SIKU  |     97.8 |     77.9 |     69.3 |     81.6 |
| Retag. (Ours) | CCR   |     97.4 |     77.5 |     70.2 |     81.7 |

* **SB**: SillokBERT (`ddokbaro/SillokBert`)
* **SIKU**: SIKU-RoBERTa (`SIKU-BERT/sikuroberta`)
* **CCR**: Classical Chinese RoBERTa (`KoichiYasuoka/roberta-classical-chinese-base-char`)

`Orig.`는 기존 annotation을 사용한 조건이고, `Retag. (Ours)`는 train/dev/test를 재태깅한 뒤 동일한 방식으로 다시 학습·평가한 조건입니다. 따라서 두 조건의 차이에는 학습 데이터와 test gold annotation 변경의 효과가 함께 포함됩니다.

주요 학습 설정은 4 epochs, learning rate `6e-5`, BF16, seed 42이며, NVIDIA RTX A6000 48GB 4개를 사용했습니다. 각 backbone은 pretrained encoder + CRF 구조로 fine-tuning했으며, 실록 dev micro F1이 가장 높은 checkpoint를 승정원일기 test 평가에 사용했습니다.

---

## 🚀 Quick Start

```bash
source .S2NER/bin/activate

export PYTHON_BIN="$PWD/.S2NER/bin/python"
export DATA_DIR="$PWD/data/retag"
export OUTPUT_ROOT="$PWD/outputs/retag"
export GPUS="0,1,2,3"
```

세 backbone을 순차적으로 학습·평가하려면:

```bash
bash baseline/run_experiments.sh all
```

특정 backbone만 실행하려면:

```bash
bash baseline/run_sillokbert.sh
bash baseline/run_siku.sh
bash baseline/run_classical_chinese.sh
```

결과는 `$OUTPUT_ROOT/{sillokbert,sikuroberta,classical_chinese}_crf/`에 저장되며, 최종 test 성능은 `test_metrics.json`에서 확인할 수 있습니다.

