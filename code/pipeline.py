

# ============================================================
# CELL 1
# Install dependencies + imports + reproducibility
# ============================================================

!pip -q install -U "pandas==2.2.2" "datasets==4.0.0" transformers accelerate evaluate shap scikit-learn pyarrow sentencepiece

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

warnings.filterwarnings("ignore")

SEED = 42

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

DEVICE = -1  # FORCE CPU ALWAYS

print("pandas version:", pd.__version__)
print("Torch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("Pipeline device:", DEVICE)

# ============================================================
# CELL 2
# Model IDs, dataset configs, and helper functions
# ============================================================

# ---------------------------
# Fixed model configuration
# ---------------------------
EN_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

AF_MODEL_ID = "Davlan/naija-twitter-sentiment-afriberta-large"
AF_MODEL_FALLBACK_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

PRIORITY_AF_LANGS = ["yor", "pcm"]  # Yoruba and Nigerian Pidgin first

# ---------------------------
# English baseline datasets
# ---------------------------
ENGLISH_DATASETS = {
    "tweet_eval": ("cardiffnlp/tweet_eval", "sentiment"),
    "imdb": ("stanfordnlp/imdb", None),
    "amazon_polarity": ("SetFit/amazon_polarity", None),
}

# ---------------------------
# African dataset configs
# ---------------------------
NAIJA_PARQUET_LANGS = {
    "hau": "hau",
    "ibo": "ibo",
    "pcm": "pcm",
    "yor": "yor",
}

NOLLY_LANG_CONFIG = {
    "eng": "en",
    "hau": "ha",
    "ibo": "ig",
    "pcm": "pcm",
    "yor": "yo",
}

YOSM_DATASET_ID = "Iyanuoluwa/YOSM"

# ---------------------------
# Helper functions
# ---------------------------
def load_any_dataset(dataset_id: str, config: Optional[str] = None):
    if config is None:
        return load_dataset(dataset_id)
    return load_dataset(dataset_id, config)

def load_naijasenti_parquet(lang: str):
    """
    Load NaijaSenti directly from converted parquet files
    to bypass deprecated dataset script loading.
    """
    lang = NAIJA_PARQUET_LANGS[lang]
    data_files = {
        "train": f"https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter/resolve/refs%2Fconvert%2Fparquet/{lang}/train/0000.parquet",
        "validation": f"https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter/resolve/refs%2Fconvert%2Fparquet/{lang}/dev/0000.parquet",
        "test": f"https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter/resolve/refs%2Fconvert%2Fparquet/{lang}/test/0000.parquet",
    }
    return load_dataset("parquet", data_files=data_files)

def load_nolly_clean(lang: str):
    cfg = NOLLY_LANG_CONFIG[lang]
    return load_dataset("Davlan/nollysenti", cfg)

def detect_text_column(columns: List[str]) -> Optional[str]:
    candidates = ["text", "tweet", "review", "sentence", "content", "comment", "document"]
    for c in candidates:
        if c in columns:
            return c
    return None

def get_label_names(ds_split) -> Optional[List[str]]:
    if "label" not in ds_split.features:
        return None
    feat = ds_split.features["label"]
    if hasattr(feat, "names") and feat.names is not None:
        return list(feat.names)
    return None

def summarize_dataset(ds_dict: DatasetDict, name: str) -> pd.DataFrame:
    rows = []
    for split_name, ds in ds_dict.items():
        row = {
            "dataset": name,
            "split": split_name,
            "n_rows": len(ds),
            "columns": ", ".join(ds.column_names),
            "text_col": detect_text_column(ds.column_names),
        }

        label_names = get_label_names(ds)
        row["label_names"] = ", ".join(label_names) if label_names else None

        if "label" in ds.column_names:
            counts = pd.Series(ds["label"]).value_counts().sort_index().to_dict()
            if label_names:
                mapped_counts = {label_names[int(k)]: int(v) for k, v in counts.items()}
                row["label_counts"] = str(mapped_counts)
            else:
                mapped_counts = {int(k): int(v) for k, v in counts.items()}
                row["label_counts"] = str(mapped_counts)
        else:
            row["label_counts"] = None

        rows.append(row)

    return pd.DataFrame(rows)

def preview_examples(ds_dict: DatasetDict, name: str, n: int = 2) -> None:
    preferred_splits = ["train", "validation", "dev", "test"]
    split_name = next((s for s in preferred_splits if s in ds_dict), None)

    if split_name is None:
        print(f"[WARN] No standard split found for {name}")
        return

    ds = ds_dict[split_name]
    text_col = detect_text_column(ds.column_names)
    cols = [c for c in [text_col, "label"] if c is not None and c in ds.column_names]

    print(f"\n{name} | preview from split='{split_name}'")
    if cols:
        display(pd.DataFrame({c: ds[:n][c] for c in cols}))
    else:
        print(ds[:n])

# def build_classifier(model_id: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#     model = AutoModelForSequenceClassification.from_pretrained(model_id)
#     clf = pipeline(
#         "text-classification",
#         model=model,
#         tokenizer=tokenizer,
#         device=DEVICE,
#         truncation=True
#     )
#     return tokenizer, model, clf

# ============================================================
# CELL 3
# Load datasets
# ============================================================

english_data = {}
african_data = {}

# ---------------------------
# Load English baseline datasets
# ---------------------------
for short_name, (dataset_id, config) in ENGLISH_DATASETS.items():
    print(f"\nLoading English dataset: {short_name}")
    english_data[short_name] = load_any_dataset(dataset_id, config=config)

# ---------------------------
# Load African datasets
# ---------------------------

# NaijaSenti via MTEB-packaged version
print("\nLoading NaijaSenti from mteb/NaijaSenti...")
try:
    african_data["naijasenti_all"] = load_dataset("mteb/NaijaSenti")
    print("[OK] Loaded mteb/NaijaSenti")
except Exception as e:
    print("[WARN] mteb/NaijaSenti failed to load:", e)
    try:
        african_data["naijasenti_all"] = load_dataset("mteb/naija_senti")
        print("[OK] Loaded mteb/naija_senti")
    except Exception as e2:
        print("[WARN] mteb/naija_senti also failed to load:", e2)

# NollySenti for priority languages
for lang in PRIORITY_AF_LANGS:
    print(f"\nLoading NollySenti: {lang}")
    try:
        african_data[f"nollysenti_{lang}"] = load_nolly_clean(lang)
        print(f"[OK] Loaded NollySenti for {lang}")
    except Exception as e:
        print(f"[WARN] Failed to load NollySenti for {lang}: {e}")

# Yoruba fallback dataset
print("\nTrying YOSM fallback...")
try:
    african_data["yosm"] = load_dataset(YOSM_DATASET_ID)
    print("[OK] Loaded YOSM")
except Exception as e:
    print("[WARN] YOSM did not load:", e)

print("\nDone loading datasets.")
print("English datasets:", list(english_data.keys()))
print("African datasets:", list(african_data.keys()))

# ============================================================
# CELL 3
# Load datasets
# ============================================================

english_data = {}
african_data = {}

# ---------------------------
# Load English baseline datasets
# ---------------------------
for short_name, (dataset_id, config) in ENGLISH_DATASETS.items():
    print(f"\nLoading English dataset: {short_name}")
    english_data[short_name] = load_any_dataset(dataset_id, config=config)

# ---------------------------
# Load African datasets
# ---------------------------

# NaijaSenti via MTEB version, per-language configs
for lang in PRIORITY_AF_LANGS:
    print(f"\nLoading NaijaSenti: {lang}")
    try:
        african_data[f"naijasenti_{lang}"] = load_dataset("mteb/NaijaSenti", lang)
        print(f"[OK] Loaded mteb/NaijaSenti for {lang}")
    except Exception as e:
        print(f"[WARN] Failed loading mteb/NaijaSenti for {lang}: {e}")
        try:
            african_data[f"naijasenti_{lang}"] = load_dataset("mteb/naija_senti", lang)
            print(f"[OK] Loaded mteb/naija_senti for {lang}")
        except Exception as e2:
            print(f"[WARN] Failed loading mteb/naija_senti for {lang}: {e2}")

# NollySenti for priority languages
for lang in PRIORITY_AF_LANGS:
    print(f"\nLoading NollySenti: {lang}")
    try:
        african_data[f"nollysenti_{lang}"] = load_nolly_clean(lang)
        print(f"[OK] Loaded NollySenti for {lang}")
    except Exception as e:
        print(f"[WARN] Failed to load NollySenti for {lang}: {e}")

print("\nDone loading datasets.")
print("English datasets:", list(english_data.keys()))
print("African datasets:", list(african_data.keys()))

# ============================================================
# CELL 4
# Quick inspection of loaded datasets
# ============================================================

for name, ds_dict in english_data.items():
    print(f"\n===== {name} =====")
    print("Splits:", list(ds_dict.keys()))
    first_split = list(ds_dict.keys())[0]
    ds = ds_dict[first_split]
    print("Columns:", ds.column_names)
    print("Features:", ds.features)
    print("First row:", ds[0])

for name, ds_dict in african_data.items():
    print(f"\n===== {name} =====")
    print("Splits:", list(ds_dict.keys()))
    first_split = list(ds_dict.keys())[0]
    ds = ds_dict[first_split]
    print("Columns:", ds.column_names)
    print("Features:", ds.features)
    print("First row:", ds[0])

# ============================================================
# CELL 5
# Standardize datasets into a common schema:
# text, label_raw, dataset_name
# ============================================================

from datasets import Dataset

def standardize_english_dataset(name, ds_dict):
    """
    Returns dict of splits with columns:
    text, label_raw, dataset_name
    """
    out = {}

    for split_name, ds in ds_dict.items():
        if name == "amazon_polarity":
            # already has text + label_text
            ds_std = ds.map(
                lambda x: {
                    "text": x["text"],
                    "label_raw": x["label_text"],
                    "dataset_name": name
                }
            )
        else:
            ds_std = ds.map(
                lambda x: {
                    "text": x["text"],
                    "label_raw": x["label"],
                    "dataset_name": name
                }
            )

        keep_cols = ["text", "label_raw", "dataset_name"]
        drop_cols = [c for c in ds_std.column_names if c not in keep_cols]
        ds_std = ds_std.remove_columns(drop_cols)
        out[split_name] = ds_std

    return out


def standardize_african_dataset(name, ds_dict):
    """
    Returns dict of splits with columns:
    text, label_raw, dataset_name
    """
    out = {}

    for split_name, ds in ds_dict.items():
        if name == "naijasenti_yor" or name == "naijasenti_pcm":
            ds_std = ds.map(
                lambda x: {
                    "text": x["text"],
                    "label_raw": x["label"],
                    "dataset_name": name
                }
            )

        elif name == "nollysenti_yor":
            ds_std = ds.map(
                lambda x: {
                    "text": x["yo_review"],
                    "label_raw": x["sentiment"],
                    "dataset_name": name
                }
            )

        elif name == "nollysenti_pcm":
            ds_std = ds.map(
                lambda x: {
                    "text": x["pcm_review"],
                    "label_raw": x["sentiment"],
                    "dataset_name": name
                }
            )
        else:
            raise ValueError(f"Unhandled African dataset: {name}")

        keep_cols = ["text", "label_raw", "dataset_name"]
        drop_cols = [c for c in ds_std.column_names if c not in keep_cols]
        ds_std = ds_std.remove_columns(drop_cols)
        out[split_name] = ds_std

    return out


english_std = {}
for name, ds_dict in english_data.items():
    english_std[name] = standardize_english_dataset(name, ds_dict)

african_std = {}
for name, ds_dict in african_data.items():
    african_std[name] = standardize_african_dataset(name, ds_dict)

print("Standardized English datasets:", list(english_std.keys()))
print("Standardized African datasets:", list(african_std.keys()))

# ============================================================
# CELL 6
# Harmonize ground-truth labels into:
# 0 = negative
# 1 = positive
# drop neutral / unknown
# ============================================================

def map_tweet_eval_label(x):
    # TweetEval: negative=0, neutral=1, positive=2
    raw = x["label_raw"]

    if raw == 0:
        return {"label": 0}   # negative
    elif raw == 2:
        return {"label": 1}   # positive
    else:
        return {"label": -1}  # neutral / unsupported


def map_imdb_label(x):
    # IMDb: negative=0, positive=1
    raw = x["label_raw"]

    if raw == 0:
        return {"label": 0}
    elif raw == 1:
        return {"label": 1}

    return {"label": -1}


def map_amazon_label(x):
    # Amazon Polarity: textual positive / negative labels
    raw = str(x["label_raw"]).strip().lower()

    if raw == "negative":
        return {"label": 0}
    elif raw == "positive":
        return {"label": 1}

    return {"label": -1}


def map_nolly_label(x):
    # NollySenti: textual positive / negative labels
    raw = str(x["label_raw"]).strip().lower()

    if raw == "negative":
        return {"label": 0}
    elif raw == "positive":
        return {"label": 1}

    return {"label": -1}


def map_naija_label(x):
    """
    NaijaSenti released class order:
    0 = positive
    1 = neutral
    2 = negative

    Harmonized binary labels:
    0 = negative
    1 = positive
    neutral is removed.
    """
    raw = x["label_raw"]

    if raw == 0:
        return {"label": 1}   # positive
    elif raw == 2:
        return {"label": 0}   # negative
    elif raw == 1:
        return {"label": -1}  # neutral

    return {"label": -1}


def apply_binary_mapping(name, ds_dict):
    out = {}

    for split_name, ds in ds_dict.items():

        if name == "tweet_eval":
            ds2 = ds.map(map_tweet_eval_label)

        elif name == "imdb":
            ds2 = ds.map(map_imdb_label)

        elif name == "amazon_polarity":
            ds2 = ds.map(map_amazon_label)

        elif name.startswith("nollysenti_"):
            ds2 = ds.map(map_nolly_label)

        elif name.startswith("naijasenti_"):
            ds2 = ds.map(map_naija_label)

        else:
            raise ValueError(f"Unhandled dataset in mapping: {name}")

        # Keep only harmonized binary labels.
        ds2 = ds2.filter(lambda x: x["label"] in [0, 1])

        out[split_name] = ds2

    return out


english_bin = {}
for name, ds_dict in english_std.items():
    english_bin[name] = apply_binary_mapping(name, ds_dict)

african_bin = {}
for name, ds_dict in african_std.items():
    african_bin[name] = apply_binary_mapping(name, ds_dict)

print("Done binary harmonization.")

# ============================================================
# CELL 7
# Check binary class balance after harmonization
# ============================================================

def show_label_counts(dataset_group, group_name):
    rows = []
    for name, ds_dict in dataset_group.items():
        for split_name, ds in ds_dict.items():
            counts = pd.Series(ds["label"]).value_counts().sort_index().to_dict()
            rows.append({
                "group": group_name,
                "dataset": name,
                "split": split_name,
                "n_rows": len(ds),
                "neg_count": counts.get(0, 0),
                "pos_count": counts.get(1, 0),
            })
    return pd.DataFrame(rows)

eng_counts = show_label_counts(english_bin, "english")
afr_counts = show_label_counts(african_bin, "african")

counts_df = pd.concat([eng_counts, afr_counts], ignore_index=True)
display(counts_df)

counts_df.to_csv("/content/binary_label_counts.csv", index=False)
print("Saved: /content/binary_label_counts.csv")

# ============================================================
# CELL 8.5
# Load models on CPU ONLY
# ============================================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def build_classifier_cpu(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)

    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=-1,
        truncation=True
    )
    return clf

print("Loading English model on CPU...")
en_clf = build_classifier_cpu("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

print("Loading African model on CPU...")
try:
    af_clf = build_classifier_cpu("Davlan/naija-twitter-sentiment-afriberta-large")
    ACTIVE_AF_MODEL = "afriberta"
    print("[OK] Afriberta loaded")
except Exception as e:
    print("[WARN] fallback model used:", e)
    af_clf = build_classifier_cpu("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual")
    ACTIVE_AF_MODEL = "xlm-roberta"

print("Models ready.")

# ============================================================
# CELL 8
# Select splits + create balanced subsets (300 pos / 300 neg)
# ============================================================

SAMPLE_SIZE_PER_CLASS = 300

def get_eval_split(ds_dict, name):
    """
    Choose consistent evaluation split
    """
    if name == "tweet_eval":
        return ds_dict["test"]
    elif name == "imdb":
        return ds_dict["test"]
    elif name == "amazon_polarity":
        return ds_dict["test"]
    elif name.startswith("nollysenti_"):
        return ds_dict["test"]
    elif name.startswith("naijasenti_"):
        return ds_dict["test"]
    else:
        raise ValueError(f"Unknown dataset: {name}")

def create_balanced_subset(ds, n_per_class=300):
    df = pd.DataFrame(ds)

    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]

    n = min(n_per_class, len(neg), len(pos))

    neg_sample = neg.sample(n=n, random_state=42)
    pos_sample = pos.sample(n=n, random_state=42)

    balanced = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=42)
    return balanced.reset_index(drop=True)

# Build evaluation datasets
eval_datasets = {}

# English
for name, ds_dict in english_bin.items():
    split_ds = get_eval_split(ds_dict, name)
    balanced_df = create_balanced_subset(split_ds, SAMPLE_SIZE_PER_CLASS)
    eval_datasets[name] = balanced_df

# African
for name, ds_dict in african_bin.items():
    split_ds = get_eval_split(ds_dict, name)
    balanced_df = create_balanced_subset(split_ds, SAMPLE_SIZE_PER_CLASS)
    eval_datasets[name] = balanced_df

print("Evaluation datasets created:")
for k, v in eval_datasets.items():
    print(k, v.shape)

# ============================================================
# CELL 8
# Select splits + create balanced subsets (300 pos / 300 neg)
# ============================================================

SAMPLE_SIZE_PER_CLASS = 300

def get_eval_split(ds_dict, name):
    """
    Choose consistent evaluation split
    """
    if name == "tweet_eval":
        return ds_dict["test"]
    elif name == "imdb":
        return ds_dict["test"]
    elif name == "amazon_polarity":
        return ds_dict["test"]
    elif name.startswith("nollysenti_"):
        return ds_dict["test"]
    elif name.startswith("naijasenti_"):
        return ds_dict["test"]
    else:
        raise ValueError(f"Unknown dataset: {name}")

def create_balanced_subset(ds, n_per_class=300):
    df = pd.DataFrame(ds)

    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]

    n = min(n_per_class, len(neg), len(pos))

    neg_sample = neg.sample(n=n, random_state=42)
    pos_sample = pos.sample(n=n, random_state=42)

    balanced = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=42)
    return balanced.reset_index(drop=True)

# Build evaluation datasets
eval_datasets = {}

# English
for name, ds_dict in english_bin.items():
    split_ds = get_eval_split(ds_dict, name)
    balanced_df = create_balanced_subset(split_ds, SAMPLE_SIZE_PER_CLASS)
    eval_datasets[name] = balanced_df

# African
for name, ds_dict in african_bin.items():
    split_ds = get_eval_split(ds_dict, name)
    balanced_df = create_balanced_subset(split_ds, SAMPLE_SIZE_PER_CLASS)
    eval_datasets[name] = balanced_df

print("Evaluation datasets created:")
for k, v in eval_datasets.items():
    print(k, v.shape)

# ============================================================
# CELL 9
# Run fixed-model inference safely
# ============================================================

def normalize_prediction_label(pred_label: str, model_family: str) -> int:
    """
    Convert model outputs to the common binary scheme:

    0 = negative
    1 = positive
    -1 = neutral / unsupported
    """
    label = str(pred_label).strip().lower()

    # --------------------------------------------------------
    # First handle explicit textual labels.
    # These are safest because they do not depend on class IDs.
    # --------------------------------------------------------
    if label in {"positive", "pos"}:
        return 1

    if label in {"negative", "neg"}:
        return 0

    if label in {"neutral", "neu"}:
        return -1

    # --------------------------------------------------------
    # English DistilBERT SST-2
    # LABEL_0 = negative
    # LABEL_1 = positive
    # --------------------------------------------------------
    if model_family == "english":
        if label in {"0", "label_0"}:
            return 0
        elif label in {"1", "label_1"}:
            return 1

    # --------------------------------------------------------
    # AfriBERTa Naija sentiment classifier
    # class order: positive, neutral, negative
    # LABEL_0 = positive
    # LABEL_1 = neutral
    # LABEL_2 = negative
    # --------------------------------------------------------
    elif model_family == "afriberta":
        if label in {"0", "label_0"}:
            return 1
        elif label in {"1", "label_1"}:
            return -1
        elif label in {"2", "label_2"}:
            return 0

    return -1

def run_inference(
    df,
    model_pipeline,
    dataset_name,
    model_family,
    batch_size=8,
    max_length=256
):
    texts = df["text"].astype(str).tolist()

    outputs = model_pipeline(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
        padding=True
    )

    pred_labels = []
    pred_scores = []

    for out in outputs:
        pred_labels.append(
            normalize_prediction_label(
                out["label"],
                model_family=model_family
            )
        )
        pred_scores.append(float(out["score"]))

    result_df = df.copy()
    result_df["pred_label"] = pred_labels
    result_df["pred_score"] = pred_scores
    result_df["dataset"] = dataset_name

    return result_df

results = {}

for name, df in eval_datasets.items():
    print(f"Running inference for: {name}")

    if name in ["tweet_eval", "imdb", "amazon_polarity"]:
        results[name] = run_inference(
            df,
            en_clf,
            name,
            model_family="english",
            batch_size=8,
            max_length=256
        )

    else:
        results[name] = run_inference(
            df,
            af_clf,
            name,
            model_family="afriberta",
            batch_size=2,
            max_length=128
        )

print("\nInference complete.")
print("results keys:", results.keys())
for k, v in results.items():
    print(k, v.shape)

print("results keys:", results.keys())

for name, df in results.items():
    print(name, df.shape)
    print("unique pred labels:", df["pred_label"].unique())

# ============================================================
# CELL 10
# Compute predictive metrics
# ============================================================

from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

metric_rows = []

for name, df in results.items():
    valid_df = df[df["pred_label"].isin([0, 1])].copy()

    if len(valid_df) == 0:
        print(f"[SKIP] {name} has no valid predictions")
        continue

    metric_rows.append({
        "dataset": name,
        "n_eval": len(valid_df),
        "accuracy": accuracy_score(valid_df["label"], valid_df["pred_label"]),
        "macro_f1": f1_score(valid_df["label"], valid_df["pred_label"], average="macro")
    })

metrics_df = pd.DataFrame(metric_rows).sort_values("dataset").reset_index(drop=True)
display(metrics_df)

metrics_df.to_csv("/content/predictive_metrics_quick.csv", index=False)
print("Saved: /content/predictive_metrics_quick.csv")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

all_results = pd.concat(results.values(), ignore_index=True)

save_path = "/content/predictions_all_datasets.csv"
all_results.to_csv(save_path, index=False)

print("Saved predictions to:", save_path)
display(all_results.head())

# ============================================================
# BOOTSTRAP CIs
# ============================================================

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

ci_rows = []

for name, df in results.items():
    valid_df = df[df["pred_label"].isin([0, 1])].copy()

    if len(valid_df) == 0:
        continue

    y_true = valid_df["label"].values
    y_pred = valid_df["pred_label"].values

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    acc_lo, acc_hi = bootstrap_metric_ci(
        y_true, y_pred, lambda yt, yp: accuracy_score(yt, yp), n_boot=1000, seed=42
    )
    f1_lo, f1_hi = bootstrap_metric_ci(
        y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro"), n_boot=1000, seed=42
    )

    ci_rows.append({
        "dataset": name,
        "n_eval": len(valid_df),
        "accuracy": round(acc, 4),
        "accuracy_ci_lower": round(acc_lo, 4),
        "accuracy_ci_upper": round(acc_hi, 4),
        "macro_f1": round(f1, 4),
        "macro_f1_ci_lower": round(f1_lo, 4),
        "macro_f1_ci_upper": round(f1_hi, 4),
    })

metrics_ci_df = pd.DataFrame(ci_rows).sort_values("dataset").reset_index(drop=True)
display(metrics_ci_df)

metrics_ci_df.to_csv("/content/predictive_metrics_with_ci.csv", index=False)
print("Saved: /content/predictive_metrics_with_ci.csv")

import os
import zipfile

ZIP_PATH = "/content/shap_results_small_FULL.zip"

with zipfile.ZipFile(ZIP_PATH, "w") as z:
    # add SHAP outputs if present
    SHAP_DIR = "/content/shap_outputs_small"
    if os.path.exists(SHAP_DIR):
        for root, dirs, files in os.walk(SHAP_DIR):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, "/content")
                z.write(full_path, arcname=arcname)
    else:
        print("No shap_outputs_small folder found.")

    # add result CSVs if present
    files = [
        "/content/binary_label_counts.csv",
        "/content/predictions_all_datasets.csv",
        "/content/predictive_metrics_quick.csv",
        "/content/predictive_metrics_with_ci.csv",
        "/content/split_half_stability.csv",
        "/content/jaccard_top20_small_shap.csv",
        "/content/final_results_table.csv",
    ]

    for f in files:
        if os.path.exists(f):
            z.write(f, arcname=os.path.basename(f))

print("Created:", ZIP_PATH)

import os
import zipfile

ZIP_PATH = "/content/shap_results_small_FULL.zip"
EXTRACT_DIR = "/content/shap_results_small_extracted_full"

os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

print("Extracted to:", EXTRACT_DIR)
print("Top-level contents:", os.listdir(EXTRACT_DIR))

import pandas as pd

all_results = pd.read_csv("/content/predictions_all_datasets.csv")
all_results.head()

SHAP_SAMPLE_SIZE = 30

target_datasets = [
    "tweet_eval",
    "imdb",
    "amazon_polarity",
    "nollysenti_yor",
    "nollysenti_pcm",
    "naijasenti_yor",
    "naijasenti_pcm",
]

shap_datasets = {}

for name in target_datasets:
    df = all_results[
        (all_results["dataset"] == name) &
        (all_results["pred_label"].isin([0, 1]))
    ].copy()

    n = min(SHAP_SAMPLE_SIZE, len(df))
    shap_datasets[name] = df.sample(n=n, random_state=42)

for k, v in shap_datasets.items():
    print(k, v.shape)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def build_classifier_cpu(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=-1,
        truncation=True
    )

en_clf = build_classifier_cpu("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

try:
    af_clf = build_classifier_cpu("Davlan/naija-twitter-sentiment-afriberta-large")
    print("Loaded Afriberta")
except Exception as e:
    print("Fallback model used:", e)
    af_clf = build_classifier_cpu("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual")

import shap

explainer_en = shap.Explainer(en_clf)
explainer_af = shap.Explainer(af_clf)

print("SHAP explainers ready")

# ============================================================
# SHAP SAVE-AS-YOU-GO LOOP (30 samples)
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

SAVE_DIR = "/content/shap_outputs_small"
os.makedirs(SAVE_DIR, exist_ok=True)

progress_path = os.path.join(SAVE_DIR, "shap_progress_small.csv")

if os.path.exists(progress_path):
    progress_df = pd.read_csv(progress_path)
    done = set(progress_df.loc[progress_df["status"] == "done", "dataset"].tolist())
    print("Resuming. Done:", done)
else:
    progress_df = pd.DataFrame(columns=["dataset", "n_texts", "status", "save_path"])
    done = set()

total = len(shap_datasets)

for i, (name, df) in enumerate(shap_datasets.items(), start=1):

    if name in done:
        print(f"[{i}/{total}] Skip: {name}")
        continue

    print(f"[{i}/{total}] SHAP: {name}")

    texts = df["text"].astype(str).tolist()
    explainer = explainer_en if name in ["tweet_eval", "imdb", "amazon_polarity"] else explainer_af

    try:
        sv = explainer(texts)

        rows = []
        for j in range(len(sv.values)):
            rows.append({
                "dataset": name,
                "row_id": j,
                "text": texts[j],
                "tokens": list(sv.data[j]),
                "shap_values": np.array(sv.values[j]).tolist()
            })

        save_path = os.path.join(SAVE_DIR, f"{name}_shap.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(rows, f)

        progress_df = pd.concat([
            progress_df,
            pd.DataFrame([{
                "dataset": name,
                "n_texts": len(texts),
                "status": "done",
                "save_path": save_path
            }])
        ], ignore_index=True)

        progress_df.to_csv(progress_path, index=False)

        print(f"Saved: {name}")

    except Exception as e:
        err_path = os.path.join(SAVE_DIR, f"{name}_error.txt")
        with open(err_path, "w") as f:
            f.write(str(e))

        progress_df = pd.concat([
            progress_df,
            pd.DataFrame([{
                "dataset": name,
                "n_texts": len(texts),
                "status": "failed",
                "save_path": err_path
            }])
        ], ignore_index=True)

        progress_df.to_csv(progress_path, index=False)

        print(f"Failed: {name} -> {e}")

print("Done SHAP run.")
display(progress_df)

# ============================================================
# LOAD SAVED SHAP FILES + BUILD GLOBAL TOKEN SCORES
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

SAVE_DIR = "/content/shap_outputs_small"

shap_saved = {}

for fname in os.listdir(SAVE_DIR):
    if fname.endswith("_shap.pkl"):
        dataset_name = fname.replace("_shap.pkl", "")
        path = os.path.join(SAVE_DIR, fname)
        with open(path, "rb") as f:
            shap_saved[dataset_name] = pickle.load(f)

print("Loaded:", list(shap_saved.keys()))

def clean_token(tok):
    tok = str(tok).strip().lower()
    if tok == "":
        return None
    if all(not ch.isalnum() for ch in tok):
        return None
    if len(tok) <= 2:
        return None
    return tok

def scalarize_shap_value(val):
    arr = np.array(val)
    if arr.ndim == 0:
        return float(arr)
    return float(np.mean(np.abs(arr)))

def aggregate_token_importance(saved_rows):
    token_scores = defaultdict(float)
    for row in saved_rows:
        for tok, val in zip(row["tokens"], row["shap_values"]):
            tok = clean_token(tok)
            if tok is None:
                continue
            token_scores[tok] += abs(scalarize_shap_value(val))
    return dict(token_scores)

global_token_scores = {
    name: aggregate_token_importance(rows)
    for name, rows in shap_saved.items()
}

print("Built global_token_scores for:", list(global_token_scores.keys()))

# ============================================================
# JACCARD + SPEARMAN
# ============================================================

def get_top_k(scores, k):
    return [tok for tok, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0.0

def get_rank_dict(scores):
    sorted_tokens = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {tok: rank for rank, (tok, _) in enumerate(sorted_tokens)}

# jaccard@20/50/100
rows = []
names = list(global_token_scores.keys())

for k in [20, 50, 100]:
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d1, d2 = names[i], names[j]
            top1 = get_top_k(global_token_scores[d1], k)
            top2 = get_top_k(global_token_scores[d2], k)

            rows.append({
                "dataset_1": d1,
                "dataset_2": d2,
                "k": k,
                "jaccard": round(jaccard(top1, top2), 4)
            })

jaccard_k_df = pd.DataFrame(rows)
display(jaccard_k_df)

# spearman
rank_dicts = {name: get_rank_dict(scores) for name, scores in global_token_scores.items()}

rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        d1, d2 = names[i], names[j]
        common = set(rank_dicts[d1]) & set(rank_dicts[d2])

        if len(common) < 5:
            corr = 0.0
        else:
            r1 = [rank_dicts[d1][t] for t in common]
            r2 = [rank_dicts[d2][t] for t in common]
            corr, _ = spearmanr(r1, r2)

        rows.append({
            "dataset_1": d1,
            "dataset_2": d2,
            "spearman_corr": round(float(corr), 4),
            "shared_tokens": len(common)
        })

spearman_df = pd.DataFrame(rows)
display(spearman_df)

jaccard_k_df.to_csv("/content/jaccard_k_full_small.csv", index=False)
spearman_df.to_csv("/content/spearman_full_small.csv", index=False)

print("Saved:")
print("/content/jaccard_k_full_small.csv")
print("/content/spearman_full_small.csv")

# ============================================================
# WITHIN-DOMAIN SPLIT-HALF STABILITY
# ============================================================

import random

def aggregate(rows):
    scores = defaultdict(float)
    for r in rows:
        for tok, val in zip(r["tokens"], r["shap_values"]):
            tok = clean_token(tok)
            if tok is None:
                continue
            scores[tok] += abs(scalarize_shap_value(val))
    return scores

rows_out = []

for name, rows in shap_saved.items():
    print(f"Processing: {name}")

    rows_copy = rows.copy()
    random.shuffle(rows_copy)

    mid = len(rows_copy) // 2
    part1 = rows_copy[:mid]
    part2 = rows_copy[mid:]

    scores1 = aggregate(part1)
    scores2 = aggregate(part2)

    top1 = get_top_k(scores1, 20)
    top2 = get_top_k(scores2, 20)

    jac = jaccard(top1, top2)

    common = set(scores1.keys()) & set(scores2.keys())
    if len(common) < 5:
        sp = 0.0
    else:
        r1 = [scores1[t] for t in common]
        r2 = [scores2[t] for t in common]
        sp, _ = spearmanr(r1, r2)

    rows_out.append({
        "dataset": name,
        "split_size": len(rows),
        "jaccard_top20_split": round(jac, 4),
        "spearman_split": round(float(sp), 4),
        "shared_tokens": len(common)
    })

split_df = pd.DataFrame(rows_out).sort_values("spearman_split", ascending=False)
display(split_df)

split_df.to_csv("/content/split_half_stability_full_small.csv", index=False)
print("Saved: /content/split_half_stability_full_small.csv")

# ============================================================
# FINAL COMBINED TABLE
# ============================================================

j20 = jaccard_k_df[jaccard_k_df["k"] == 20].copy()
j20 = j20.rename(columns={"jaccard": "jaccard_top20"})

def summarize_pairwise(df, col):
    out = []
    for d in metrics_ci_df["dataset"]:
        vals = [
            r[col] for _, r in df.iterrows()
            if r["dataset_1"] == d or r["dataset_2"] == d
        ]
        out.append({
            "dataset": d,
            f"{col}_avg": round(float(np.mean(vals)), 4) if vals else None
        })
    return pd.DataFrame(out)

j20_avg = summarize_pairwise(j20, "jaccard_top20")
sp_avg = summarize_pairwise(spearman_df, "spearman_corr")

final_df = (
    metrics_ci_df
    .merge(split_df[["dataset", "jaccard_top20_split", "spearman_split"]], on="dataset", how="left")
    .merge(j20_avg, on="dataset", how="left")
    .merge(sp_avg, on="dataset", how="left")
)

display(final_df)
final_df.to_csv("/content/final_results_table.csv", index=False)
print("Saved: /content/final_results_table.csv")

import os, zipfile

ZIP_PATH = "/content/paper_results_final.zip"

files = [
    "/content/final_results_table.csv",
    "/content/predictive_metrics_with_ci.csv",
    "/content/jaccard_k_full_small.csv",
    "/content/spearman_full_small.csv",
    "/content/split_half_stability_full_small.csv",
    "/content/predictions_all_datasets.csv",
]

with zipfile.ZipFile(ZIP_PATH, "w") as z:
    for f in files:
        if os.path.exists(f):
            z.write(f, arcname=os.path.basename(f))

print("Created:", ZIP_PATH)

import os
import zipfile

ZIP_PATH = "/content/paper_results_final.zip"

files = [
    "/content/final_results_table.csv",
    "/content/predictive_metrics_with_ci.csv",
    "/content/jaccard_k_full_small.csv",
    "/content/spearman_full_small.csv",
    "/content/split_half_stability_full_small.csv",
    "/content/predictions_all_datasets.csv",
]

with zipfile.ZipFile(ZIP_PATH, "w") as z:
    for f in files:
        if os.path.exists(f):
            z.write(f, arcname=os.path.basename(f))

print("Created:", ZIP_PATH)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/predictive_metrics_with_ci.csv").sort_values("accuracy", ascending=False)

plt.figure(figsize=(9, 5))
plt.errorbar(
    df["dataset"],
    df["accuracy"],
    yerr=[
        df["accuracy"] - df["accuracy_ci_lower"],
        df["accuracy_ci_upper"] - df["accuracy"]
    ],
    fmt="o",
    capsize=4
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Predictive Performance Across Datasets (95% CI)")
plt.tight_layout()
plt.savefig("/content/figure_accuracy_ci.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_accuracy_ci.png")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

jdf = pd.read_csv("/content/jaccard_k_full_small.csv")
j20 = jdf[jdf["k"] == 20].copy()

datasets = sorted(set(j20["dataset_1"]).union(set(j20["dataset_2"])))
mat = pd.DataFrame(np.nan, index=datasets, columns=datasets)

for d in datasets:
    mat.loc[d, d] = 1.0

for _, r in j20.iterrows():
    d1, d2, v = r["dataset_1"], r["dataset_2"], r["jaccard"]
    mat.loc[d1, d2] = v
    mat.loc[d2, d1] = v

plt.figure(figsize=(8, 6))
plt.imshow(mat.fillna(0).values, aspect="auto")
plt.xticks(range(len(datasets)), datasets, rotation=45, ha="right")
plt.yticks(range(len(datasets)), datasets)
plt.colorbar(label="Jaccard@20")
plt.title("Cross-Domain SHAP Token Overlap")
plt.tight_layout()
plt.savefig("/content/figure_jaccard20_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_jaccard20_heatmap.png")

import pandas as pd

df = pd.read_csv("/content/final_results_table.csv")

round_cols = [
    "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
    "macro_f1", "macro_f1_ci_lower", "macro_f1_ci_upper",
    "jaccard_top20_split", "spearman_split",
    "jaccard_top20_avg", "spearman_corr_avg"
]

for c in round_cols:
    df[c] = df[c].round(3)

df.to_csv("/content/final_results_table_rounded.csv", index=False)
print("Saved: /content/final_results_table_rounded.csv")
display(df)

import os, zipfile

ZIP_PATH = "/content/paper_artifacts_final.zip"

files = [
    "/content/final_results_table.csv",
    "/content/final_results_table_rounded.csv",
    "/content/predictive_metrics_with_ci.csv",
    "/content/jaccard_k_full_small.csv",
    "/content/spearman_full_small.csv",
    "/content/split_half_stability_full_small.csv",
    "/content/predictions_all_datasets.csv",
    "/content/figure_accuracy_ci.png",
    "/content/figure_jaccard20_heatmap.png",
]

with zipfile.ZipFile(ZIP_PATH, "w") as z:
    for f in files:
        if os.path.exists(f):
            z.write(f, arcname=os.path.basename(f))

    shap_dir = "/content/shap_outputs_small"
    if os.path.exists(shap_dir):
        for root, dirs, fs in os.walk(shap_dir):
            for f in fs:
                full = os.path.join(root, f)
                z.write(full, arcname=os.path.relpath(full, "/content"))

print("Created:", ZIP_PATH)

from google.colab import files
files.download("/content/paper_artifacts_final.zip")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/jaccard_k_full_small.csv")

pairs_to_plot = [
    ("amazon_polarity", "imdb"),
    ("imdb", "tweet_eval"),
    ("naijasenti_pcm", "nollysenti_pcm"),
    ("naijasenti_yor", "nollysenti_yor"),
]

plt.figure(figsize=(8,5))

for d1, d2 in pairs_to_plot:
    sub = df[
        ((df["dataset_1"] == d1) & (df["dataset_2"] == d2)) |
        ((df["dataset_1"] == d2) & (df["dataset_2"] == d1))
    ].sort_values("k")
    if len(sub) > 0:
        plt.plot(sub["k"], sub["jaccard"], marker="o", label=f"{d1} vs {d2}")

plt.xlabel("k")
plt.ylabel("Jaccard")
plt.title("Cross-Domain SHAP Token Overlap at Different k")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/content/figure_jaccard_k_lines.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_jaccard_k_lines.png")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/content/final_results_table.csv")

plot_df = df[[
    "dataset",
    "jaccard_top20_split",
    "jaccard_top20_avg",
    "spearman_split",
    "spearman_corr_avg"
]].copy()

# Jaccard chart
x = np.arange(len(plot_df))
w = 0.35

plt.figure(figsize=(9,5))
plt.bar(x - w/2, plot_df["jaccard_top20_split"], width=w, label="Within-domain")
plt.bar(x + w/2, plot_df["jaccard_top20_avg"], width=w, label="Cross-domain avg")
plt.xticks(x, plot_df["dataset"], rotation=45, ha="right")
plt.ylabel("Jaccard@20")
plt.title("Within-Domain vs Cross-Domain SHAP Stability")
plt.legend()
plt.tight_layout()
plt.savefig("/content/figure_within_vs_cross_jaccard.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_within_vs_cross_jaccard.png")

import pandas as pd
import os

manifest = pd.DataFrame({
    "file": [
        "final_results_table.csv",
        "final_results_table_rounded.csv",
        "predictive_metrics_with_ci.csv",
        "jaccard_k_full_small.csv",
        "spearman_full_small.csv",
        "split_half_stability_full_small.csv",
        "predictions_all_datasets.csv",
        "figure_accuracy_ci.png",
        "figure_jaccard20_heatmap.png",
        "figure_jaccard_k_lines.png",
        "figure_within_vs_cross_jaccard.png",
    ]
})

manifest["exists"] = manifest["file"].apply(lambda x: os.path.exists(f"/content/{x}"))
manifest.to_csv("/content/results_manifest.csv", index=False)
print("Saved: /content/results_manifest.csv")
display(manifest)

import pandas as pd
import numpy as np

df = pd.read_csv("/content/predictions_all_datasets.csv")

eda_rows = []

for name, g in df.groupby("dataset"):
    text_lengths = g["text"].astype(str).str.len()
    token_lengths = g["text"].astype(str).str.split().str.len()

    eda_rows.append({
        "dataset": name,
        "n_rows": len(g),
        "n_valid_preds": int(g["pred_label"].isin([0, 1]).sum()),
        "avg_char_len": round(text_lengths.mean(), 1),
        "median_char_len": round(text_lengths.median(), 1),
        "avg_token_len": round(token_lengths.mean(), 1),
        "median_token_len": round(token_lengths.median(), 1),
        "neg_rate": round((g["label"] == 0).mean(), 3),
        "pos_rate": round((g["label"] == 1).mean(), 3),
    })

eda_df = pd.DataFrame(eda_rows).sort_values("dataset").reset_index(drop=True)
display(eda_df)

eda_df.to_csv("/content/dataset_eda_summary.csv", index=False)
print("Saved: /content/dataset_eda_summary.csv")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/predictions_all_datasets.csv")
df["token_len"] = df["text"].astype(str).str.split().str.len()

plot_df = df.groupby("dataset")["token_len"].mean().sort_values(ascending=False)

plt.figure(figsize=(9,5))
plot_df.plot(kind="bar")
plt.ylabel("Average token length")
plt.title("Average Text Length by Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/content/figure_avg_text_length.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_avg_text_length.png")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/predictions_all_datasets.csv")
valid_df = df[df["pred_label"].isin([0, 1])].copy()

conf_df = valid_df.groupby("dataset")["pred_score"].mean().sort_values(ascending=False)

plt.figure(figsize=(9,5))
conf_df.plot(kind="bar")
plt.ylabel("Average predicted confidence")
plt.title("Average Prediction Confidence by Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/content/figure_avg_confidence.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_avg_confidence.png")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/predictions_all_datasets.csv")
valid_df = df[df["pred_label"].isin([0, 1])].copy()
valid_df["is_correct"] = (valid_df["label"] == valid_df["pred_label"]).astype(int)

err_df = (1 - valid_df.groupby("dataset")["is_correct"].mean()).sort_values(ascending=False)

plt.figure(figsize=(9,5))
err_df.plot(kind="bar")
plt.ylabel("Error rate")
plt.title("Prediction Error Rate by Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/content/figure_error_rate.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: /content/figure_error_rate.png")

# ============================================================
# ZIP ONLY RESULT / FIGURE / CSV / PKL FILES + DOWNLOAD
# ============================================================

import os
import zipfile
from google.colab import files

ZIP_PATH = "/content/all_results_only.zip"

allowed_ext = {".csv", ".png", ".jpg", ".jpeg", ".pkl", ".txt", ".zip", ".md", ".pdf"}

folders_to_include = [
    "/content/shap_outputs_small",
]

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for item in os.listdir("/content"):
        full_path = os.path.join("/content", item)

        if os.path.isfile(full_path) and item != os.path.basename(ZIP_PATH):
            ext = os.path.splitext(item)[1].lower()
            if ext in allowed_ext:
                z.write(full_path, arcname=item)

    for folder in folders_to_include:
        if os.path.exists(folder):
            for root, dirs, fs in os.walk(folder):
                for f in fs:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in allowed_ext:
                        full_path = os.path.join(root, f)
                        arcname = os.path.relpath(full_path, "/content")
                        z.write(full_path, arcname=arcname)

print("Created:", ZIP_PATH)
files.download(ZIP_PATH)

# ============================================================
# FIGURE: Within-domain vs cross-domain Jaccard
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

final_df = pd.read_csv("/content/final_results_table.csv")

order = [
    "amazon_polarity",
    "imdb",
    "tweet_eval",
    "nollysenti_pcm",
    "nollysenti_yor",
    "naijasenti_pcm",
    "naijasenti_yor",
]

final_df["dataset"] = pd.Categorical(final_df["dataset"], categories=order, ordered=True)
final_df = final_df.sort_values("dataset").reset_index(drop=True)

x = np.arange(len(final_df))
width = 0.38

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, final_df["jaccard_top20_split"], width=width, label="Within-domain Jaccard@20")
plt.bar(x + width/2, final_df["jaccard_top20_avg"], width=width, label="Cross-domain avg Jaccard@20")

plt.xticks(x, final_df["dataset"], rotation=45, ha="right")
plt.ylabel("Jaccard@20")
plt.title("Within-Domain vs Cross-Domain SHAP Stability")
plt.ylim(0, 0.35)
plt.legend()
plt.tight_layout()

save_path = "/content/figure_within_vs_cross_jaccard.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)

# ============================================================
# FIGURE: Split-half stability by dataset
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load if needed
split_df = pd.read_csv("/content/split_half_stability_full_small.csv")

# optional: nicer order
order = [
    "amazon_polarity",
    "imdb",
    "tweet_eval",
    "nollysenti_pcm",
    "nollysenti_yor",
    "naijasenti_pcm",
    "naijasenti_yor",
]

split_df["dataset"] = pd.Categorical(split_df["dataset"], categories=order, ordered=True)
split_df = split_df.sort_values("dataset").reset_index(drop=True)

x = np.arange(len(split_df))
width = 0.38

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, split_df["jaccard_top20_split"], width=width, label="Jaccard@20 (split-half)")
plt.bar(x + width/2, split_df["spearman_split"], width=width, label="Spearman (split-half)")

plt.xticks(x, split_df["dataset"], rotation=45, ha="right")
plt.ylabel("Stability score")
plt.title("Within-Domain SHAP Stability (Split-Half)")
plt.ylim(0, 0.65)
plt.legend()
plt.tight_layout()

save_path = "/content/figure_split_half_stability.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)

# ============================================================
# FIGURE 1: Accuracy with 95% CI
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/predictive_metrics_with_ci.csv")

# sort for nicer display
df = df.sort_values("accuracy", ascending=False)

plt.figure(figsize=(9,5))

plt.bar(df["dataset"], df["accuracy"])

# error bars
plt.errorbar(
    df["dataset"],
    df["accuracy"],
    yerr=[
        df["accuracy"] - df["accuracy_ci_lower"],
        df["accuracy_ci_upper"] - df["accuracy"]
    ],
    fmt="none",
    capsize=4
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Predictive Performance Across Datasets (95% CI)")
plt.tight_layout()

save_path = "/content/figure_accuracy_ci.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)

# ============================================================
# FIGURE 2: Jaccard@20 heatmap
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

jdf = pd.read_csv("/content/jaccard_k_full_small.csv")
j20 = jdf[jdf["k"] == 20]

datasets = sorted(set(j20["dataset_1"]).union(set(j20["dataset_2"])))
mat = pd.DataFrame(np.nan, index=datasets, columns=datasets)

# fill diagonal
for d in datasets:
    mat.loc[d, d] = 1.0

# fill values
for _, r in j20.iterrows():
    d1, d2, v = r["dataset_1"], r["dataset_2"], r["jaccard"]
    mat.loc[d1, d2] = v
    mat.loc[d2, d1] = v

plt.figure(figsize=(8,6))
plt.imshow(mat.fillna(0).values, aspect="auto")

plt.xticks(range(len(datasets)), datasets, rotation=45, ha="right")
plt.yticks(range(len(datasets)), datasets)

plt.colorbar(label="Jaccard@20")
plt.title("Cross-Domain SHAP Token Overlap (Jaccard@20)")
plt.tight_layout()

save_path = "/content/figure_jaccard20_heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)

# ============================================================
# FIGURE 3: Jaccard vs K
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/jaccard_k_full_small.csv")

pairs = [
    ("amazon_polarity", "imdb"),
    ("imdb", "tweet_eval"),
    ("naijasenti_pcm", "nollysenti_pcm"),
    ("naijasenti_yor", "nollysenti_yor"),
]

plt.figure(figsize=(8,5))

for d1, d2 in pairs:
    sub = df[
        ((df["dataset_1"] == d1) & (df["dataset_2"] == d2)) |
        ((df["dataset_1"] == d2) & (df["dataset_2"] == d1))
    ].sort_values("k")

    if len(sub) > 0:
        plt.plot(sub["k"], sub["jaccard"], marker="o", label=f"{d1} vs {d2}")

plt.xlabel("k")
plt.ylabel("Jaccard")
plt.title("Cross-Domain SHAP Token Overlap vs Top-k")
plt.legend(fontsize=8)
plt.tight_layout()

save_path = "/content/figure_jaccard_k_lines.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)

# Save EVERYTHING
import shutil

shutil.make_archive("/content/final_backup", 'zip', "/content")
