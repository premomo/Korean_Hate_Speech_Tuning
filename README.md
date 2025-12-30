# KMHaS Korean Hate Speech Multi‑Label Classification

한국어 혐오 발언(Korean Hate Speech)을 **멀티라벨 분류(Multi‑Label Classification)** 방식으로 탐지하는 파인튜닝 프로젝트입니다.
FastCampus AI Labs 과제 및 개인 실험용으로 진행되었으며, **KMHaS 데이터셋 + ELECTRA 계열 모델**을 기반으로 합니다.

---

## 📌 Git Repository
https://github.com/premomo/Korean_Hate_Speech_Tuning
---

## 📌 Project Overview

* **Task**: Korean Hate Speech Multi‑Label Classification
* **Dataset**: KMHaS (jeanlee/kmhas_korean_hate_speech)
* **Base Models**:

  * `beomi/KcELECTRA-base-v2022`
  * `monologg/koelectra`
  * `snunlp/KR-Medium`
* **Framework**: PyTorch + HuggingFace Transformers
* **Loss**: `BCEWithLogitsLoss`
* **Evaluation**: Micro / Macro F1, Accuracy (flatten 기준)

---

## 🧠 Labels (8‑Class Multi‑Label)

| Index | English   | Korean    |
| ----: | --------- | --------- |
|     0 | origin    | 출신/이주민 혐오 |
|     1 | physical  | 외모 비하     |
|     2 | politics  | 정치/이념 혐오  |
|     3 | profanity | 일반 욕설     |
|     4 | age       | 연령 비하     |
|     5 | gender    | 성별 혐오     |
|     6 | race      | 인종 혐오     |
|     7 | religion  | 종교 혐오     |

> 하나의 문장에 **복수 라벨이 동시에 활성화**될 수 있습니다.

---

## 📂 Repository Structure

```
.
├─ train_kmhas_multilabel_v5.py     # 메인 학습/추론 스크립트
├─ Model_Compare.ipynb             # 모델별 성능 비교
├─ Multi_label_fine_tuning_*.ipynb # 모델별 실험 노트북
├─ project/
│  └─ kmhas_kcelectra_multilabel_v5/
│     ├─ config.json
│     ├─ pytorch_model.bin
│     ├─ tokenizer.json
│     └─ meta.json                 # 실험 메타데이터 & threshold
└─ README.md
```

---

## ⚙️ Training Configuration (v5)

```text
Model           : beomi/KcELECTRA-base-v2022
Epochs          : 4
Learning Rate   : 3e-5
Train Batch     : 32
Eval Batch      : 32
Max Length      : 128
Scheduler       : Cosine + Warmup
Warmup Ratio    : 0.1
Seed            : 42
```

* **LR Scheduler**: `cosine` (linear로 교체 가능)
* **Threshold Tuning**: Label‑wise F1 최적화

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install torch transformers datasets scikit-learn
```

### 2️⃣ Train Model

```python
DO_TRAIN = True
python train_kmhas_multilabel_v5.py
```

### 3️⃣ Load & Inference

```python
DO_TRAIN = False
python train_kmhas_multilabel_v5.py
```

---

## 🔍 Inference Example

```python
text = "한국 남자들은 너무 혐오스럽다"
detected, latency_ms = infer(text)
```

Output:

```
검출 라벨:
- 성별 혐오 : 0.91
- 일반 욕설 : 0.84
추론 시간(ms): 12.3
```

> Threshold는 validation 기준으로 **라벨별 개별 튜닝**되어 적용됩니다.

---

## 📊 Evaluation Strategy

* **Flatten 방식 평가**

  * 모든 라벨을 1D 벡터로 펼쳐 Micro / Macro F1 계산
* **Baseline**: threshold = 0.5
* **Improved**: threshold grid search (0.05 ~ 0.95)

`meta.json`에 다음 정보가 자동 저장됩니다:

* 실험 설정
* Best epoch metrics
* Tuned threshold
* Tuned metrics

---

## 🧪 Experiments

* v1 ~ v3: LR / Epoch 비교
* v4: Batch size 64 (속도 이슈로 중단)
* **v5**: Batch 32 + Cosine Scheduler + Threshold Tuning

---

## 🎯 Notes

* 혐오 발언 탐지는 **문맥 의존적**이며, 데이터 특성상 false positive 가능성이 있습니다.
* 실제 서비스 적용 시:

  * rule‑based filter 병행
  * confidence score 기반 UX 설계 권장

---

## 📜 License & Disclaimer

* Dataset: KMHaS (연구/교육 목적)
* 본 프로젝트는 **연구 및 학습 목적**이며, 실제 서비스 적용 시 추가 검증이 필요합니다.

---

## 👤 Author

* **Fast Campus Team 2**
  FastCampus AI Labs



