# S2NER

**S2NER**은 **조선왕조실록(Joseon Wangjo Sillok)**과 **승정원일기(Seungjeongwon Ilgi)**를 대상으로 한 고전 한문 문자 단위 개체명 인식(NER) baseline입니다.

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
├── S2NERdictionaries/
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

