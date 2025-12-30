import os
import json
import time
import random
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)

# ============================================================
# 0) 설정 (v5 = v3 기반 + LR scheduler 적용)
# ============================================================
MODEL_NAME = "beomi/KcELECTRA-base-v2022"

RUN_NAME = "v5_lr_3e-5_cosine_warmup"
CHANGE_NOTE = "v4(train_bs=64) 너무 느려서 v3(train_bs=32)로 복귀 + cosine lr scheduler(warmup) 적용"

SAVE_DIR = "./project/kmhas_kcelectra_multilabel_v5"
META_PATH = os.path.join(SAVE_DIR, "meta.json")

DO_TRAIN = False
SEED = 42

EPOCHS = 4
LR = 3e-5

MAX_LEN = 128
TRAIN_BS = 32
EVAL_BS = 32

LOG_EVERY = 200
THR_GRID = np.arange(0.05, 0.96, 0.05)

NUM_WORKERS = 0
PIN_MEMORY = True

# ✅ Scheduler 설정
LR_SCHEDULER_TYPE = "cosine"   # "linear" 로 바꾸면 linear warmup+decay
WARMUP_RATIO = 0.1            # 0.05~0.1 추천

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
print("cuda available:", torch.cuda.is_available())
print("RUN:", RUN_NAME)
print("NOTE:", CHANGE_NOTE)
print("CONFIG:", f"epochs={EPOCHS}, lr={LR}, train_bs={TRAIN_BS}, eval_bs={EVAL_BS}, max_len={MAX_LEN}")
print("SCHEDULER:", f"type={LR_SCHEDULER_TYPE}, warmup_ratio={WARMUP_RATIO}")

# ============================================================
# 0-1) 재현성 고정
# ============================================================
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

set_seed(SEED)

# ============================================================
# 1) 데이터 로드 + 라벨 정의
# ============================================================
print("📦 Loading KMHaS dataset...")
ds = load_dataset("jeanlee/kmhas_korean_hate_speech")
print("✅ example:", ds["train"][0])

LABELS_EN = ["origin", "physical", "politics", "profanity", "age", "gender", "race", "religion"]
LABELS_KO = [
    "출신/이주민 혐오",
    "외모 비하",
    "정치/이념 혐오",
    "일반 욕설",
    "연령 비하",
    "성별 혐오",
    "인종 혐오",
    "종교 혐오",
]
num_labels = len(LABELS_EN)

print("✅ num_labels:", num_labels)
print("✅ LABELS_EN:", LABELS_EN)

# ============================================================
# 2) 토크나이저/콜레이터
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# ============================================================
# 3) Dataset -> multi-hot + tokenize
# ============================================================
def to_multihot(example):
    y = np.zeros(num_labels, dtype=np.float32)
    for idx in example["label"]:
        if idx == 8:
            continue
        if 0 <= idx < 8:
            y[idx] = 1.0
    example["labels"] = y
    return example

def tokenize(batch):
    tok = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    tok["labels"] = batch["labels"]
    return tok

def build_loaders():
    ds2 = ds.map(to_multihot, desc="to_multihot")

    remove_cols = ds2["train"].column_names
    ds_tok = ds2.map(tokenize, batched=True, remove_columns=remove_cols, desc="tokenize")
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        ds_tok["train"],
        batch_size=TRAIN_BS,
        shuffle=True,
        generator=g,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        ds_tok["validation"],
        batch_size=EVAL_BS,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=data_collator
    )
    return train_loader, val_loader

# ============================================================
# 4) metrics/threshold 튜닝
# ============================================================
def collect_probs_and_labels(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(batch["labels"].cpu().numpy().astype(int))
    return np.vstack(probs_all), np.vstack(labels_all)

def eval_metrics_from_probs_flatten(probs, labels, thr_vec):
    from sklearn.metrics import f1_score, accuracy_score
    preds = (probs >= thr_vec.reshape(1, -1)).astype(int)

    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)

    micro_f1 = f1_score(labels_flat, preds_flat, average="micro", zero_division=0)
    macro_f1 = f1_score(labels_flat, preds_flat, average="macro", zero_division=0)
    acc = accuracy_score(labels_flat, preds_flat)

    return float(micro_f1), float(macro_f1), float(acc)

def tune_thresholds_per_label(probs, labels, grid):
    from sklearn.metrics import f1_score
    best_thr = np.full(labels.shape[1], 0.5, dtype=np.float32)

    for i in range(labels.shape[1]):
        y_true = labels[:, i]
        if y_true.sum() == 0:
            best_thr[i] = 0.5
            continue

        best_f1, best_t = -1.0, 0.5
        for t in grid:
            y_pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        best_thr[i] = best_t

    return best_thr

# ============================================================
# 5) 메타 저장/로드
# ============================================================
def save_meta(thresholds, best_epoch_metrics, tuned_metrics):
    os.makedirs(SAVE_DIR, exist_ok=True)
    meta = {
        "run_name": RUN_NAME,
        "change_note": CHANGE_NOTE,
        "model_name": MODEL_NAME,
        "seed": SEED,

        "num_labels": num_labels,
        "labels_en": LABELS_EN,
        "labels_ko": LABELS_KO,

        "max_len": MAX_LEN,
        "lr": LR,
        "epochs": EPOCHS,
        "train_bs": TRAIN_BS,
        "eval_bs": EVAL_BS,
        "loss": "BCEWithLogitsLoss",
        "weight_decay": 0.01,

        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_ratio": WARMUP_RATIO,

        "threshold_grid": [float(x) for x in THR_GRID],
        "thresholds": [float(x) for x in thresholds],

        "best_epoch_metrics_at_0.5_flatten": {
            "val_micro_f1": best_epoch_metrics[0],
            "val_macro_f1": best_epoch_metrics[1],
            "val_acc": best_epoch_metrics[2],
        },
        "tuned_metrics_flatten": {
            "val_micro_f1": tuned_metrics[0],
            "val_macro_f1": tuned_metrics[1],
            "val_acc": tuned_metrics[2],
        }
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 6) 학습 + best 저장 + threshold 튜닝 + meta 저장
# ============================================================
def train_and_save():
    train_loader, val_loader = build_loaders()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    # ✅ LR Scheduler 생성
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print("✅ Scheduler ready:",
          f"type={LR_SCHEDULER_TYPE}, warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")

    best_micro_f1 = -1.0
    best_state = None
    best_epoch_metrics = (0.0, 0.0, 0.0)

    thr_05 = np.full(num_labels, 0.5, dtype=np.float32)

    print("🚀 Training start...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).logits

            loss = loss_fn(logits, batch["labels"].float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

            if step % LOG_EVERY == 0:
                curr_lr = lr_scheduler.get_last_lr()[0]
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {total_loss/step:.4f} | lr {curr_lr:.8f}")

        val_probs, val_labels = collect_probs_and_labels(model, val_loader)
        val_micro_f1, val_macro_f1, val_acc = eval_metrics_from_probs_flatten(val_probs, val_labels, thr_05)

        train_loss = total_loss / len(train_loader)
        print(
            f"✅ epoch {epoch} done | "
            f"train_loss {train_loss:.4f} | "
            f"val_micro_f1@0.5 {val_micro_f1:.4f} | "
            f"val_macro_f1@0.5 {val_macro_f1:.4f} | "
            f"val_acc@0.5 {val_acc:.4f}"
        )

        if val_micro_f1 > best_micro_f1:
            best_micro_f1 = val_micro_f1
            best_epoch_metrics = (val_micro_f1, val_macro_f1, val_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("🎯 Tuning per-label thresholds on validation set...")
    val_probs, val_labels = collect_probs_and_labels(model, val_loader)
    thresholds = tune_thresholds_per_label(val_probs, val_labels, THR_GRID)

    tuned_micro_f1, tuned_macro_f1, tuned_acc = eval_metrics_from_probs_flatten(val_probs, val_labels, thresholds)
    print(
        f"✅ tuned metrics (flatten) | "
        f"val_micro_f1 {tuned_micro_f1:.4f} | "
        f"val_macro_f1 {tuned_macro_f1:.4f} | "
        f"val_acc {tuned_acc:.4f}"
    )

    print("✅ thresholds (KO):")
    for i in range(num_labels):
        print(i, LABELS_KO[i], "->", round(float(thresholds[i]), 2))

    save_meta(thresholds, best_epoch_metrics, (tuned_micro_f1, tuned_macro_f1, tuned_acc))
    print("💾 Saved finetuned model to:", SAVE_DIR)

    return model, thresholds

# ============================================================
# 7) 로드
# ============================================================
def load_finetuned():
    if not os.path.exists(SAVE_DIR) or not os.path.exists(META_PATH):
        raise RuntimeError("저장된 모델이 없습니다. DO_TRAIN=True로 한 번 학습/저장부터 해주세요.")

    meta = load_meta()
    print("📌 Loaded meta keys:", list(meta.keys()))
    print("RUN_NAME in meta:", meta.get("run_name"), "| NOTE:", meta.get("change_note"))

    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
    model.eval()

    thresholds = np.array(meta.get("thresholds", [0.5] * num_labels), dtype=np.float32)
    labels_ko = meta.get("labels_ko", LABELS_KO)
    return model, thresholds, labels_ko

# ============================================================
# 8) 실행 분기
# ============================================================
if DO_TRAIN:
    v5_model, thresholds = train_and_save()
    labels_ko = LABELS_KO
else:
    v5_model, thresholds, labels_ko = load_finetuned()

# ============================================================
# 9) 단독 추론 출력 (v5)
# ============================================================
def predict_probs(model, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def infer(text):
    t0 = time.time()
    probs = predict_probs(v5_model, text)

    detected = [(labels_ko[i], float(probs[i])) for i in range(num_labels) if probs[i] >= thresholds[i]]
    detected.sort(key=lambda x: x[1], reverse=True)

    ms = (time.time() - t0) * 1000
    return detected, ms

# tests = [
#     "오늘 날씨가 좋아서 산책하러 나갔다 왔어요.",
#     "와 진짜 오늘 일 개빡셌다, 나 완전 녹초 됐어.",
#     "여자들은 감정적이라서 중요한 일에는 맡기면 안 된다.",
#     "이민자들은 다 쫓아내야 나라가 산다.",
#     "댓글에 “이민자들은 다 쫓아내야 한다”라는 글이 올라와서 충격 받았다.",
# ]

tests = [
    "한국 남자들은 너무 혐오스럽다",
]

print("\n\n==================== (v5) 단독 멀티라벨 출력 ====================")
for t in tests:
    detected, ms = infer(t)
    print("\n" + "="*100)
    print("[입력 문장]")
    print(t)
    print("-"*100)
    print("추론 시간(ms):", round(ms, 2))
    if not detected:
        print("검출 라벨: 없음")
    else:
        print("검출 라벨:")
        for name, p in detected:
            print("-", name, ":", round(p, 4))
    print("="*100)
