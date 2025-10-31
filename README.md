# 11785_group6
> 11785 group project.

1. dpo_toxic-main: baseline implementation 1 from https://github.com/ajyl/dpo_toxic

2. DPO_LoRA_Baseline.ipynb: baseline implementation 2 from https://github.com/mitultiwari/DPO_Project


## Directory Structure
ASC/
├── baseline/
│ ├── baseline.py 
│ └── config.py
├── model/
└── utils/


## Baseline code
`python ASC/baseline/baseline.py`


## Dataset
TRAIN split:
  - Total samples: 160,800
  - Sanity_check : 10,000

TEST split:
  - Total samples: 8,552
  - Evaluation(while training) : 1,000
  - Toxicity test: 50


