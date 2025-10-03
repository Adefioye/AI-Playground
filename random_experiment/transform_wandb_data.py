from pathlib import Path
import re
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
from caas_jupyter_tools import display_dataframe_to_user

# --- Inputs ---
csv_path = Path("./wandb_export_2025-10-03T09_56_38.127-05_00.csv")  # path to uploaded file
OUTPUT_CSV = Path("./full_tasks_results.csv")

# Required tasks list
TASKS = [
    "hellaswag",
    "lambada_openai",
    "mmlu",
    "piqa",
    "wikitext",
    "winogrande",
    "commonsense_qa",
    "pubmedqa",
    "race",
    "sciq",
    "wsc273",
    "xnli",
]

# Model tokens to normalize names
MODEL_TOKENS = [
    "gpt2",
    "mha",
    "mla192-96-0",
    "mla192-96-192",
    "mla0-96-192",
    "mla0-0-192",
    "mla0-128-0",
    "mla192-0-0",
    "mla0-0-0",
]


def load_wandb_export(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def find_model_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if df[c].astype(str).str.contains("full-tasks", case=False, na=False).any()]
    if not candidates:
        return None
    counts = {c: df[c].astype(str).str.contains("full-tasks", case=False, na=False).sum() for c in candidates}
    return max(counts, key=counts.get)


def normalize_model_name(raw: str) -> Optional[str]:
    raw_l = str(raw).lower()
    matches = [tok for tok in MODEL_TOKENS if tok in raw_l]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


def select_metric_columns(df: pd.DataFrame, tasks: List[str]) -> List[str]:
    cols = []
    for col in df.columns:
        if "/" not in col:
            continue
        task, metric = col.split("/", 1)
        if task in tasks:
            cols.append(col)
    return cols


def pretty_row_label(col_name: str) -> str:
    task, metric = col_name.split("/", 1)
    return f"{task}\n({metric})"


def format_acc_like(col: pd.Series) -> pd.Series:
    # metric name inside parentheses from the index label
    metric = col.name.split("\n(")[-1].rstrip(") ")
    if metric in {"acc", "acc_norm"}:
        return pd.to_numeric(col, errors="coerce").mul(100).round(2)
    return col


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    model_col = find_model_column(df)
    if model_col is None:
        raise RuntimeError("Could not find a column with 'full-tasks' to identify models.")
    df_ft = df[df[model_col].astype(str).str.contains("full-tasks", case=False, na=False)].copy()
    if df_ft.empty:
        raise RuntimeError("No rows with 'full-tasks' found.")
    df_ft["__model__"] = df_ft[model_col].apply(normalize_model_name)
    df_ft = df_ft.dropna(subset=["__model__"])
    if df_ft.empty:
        raise RuntimeError("No recognizable model tokens among 'full-tasks' rows.")
    metric_cols = select_metric_columns(df_ft, TASKS)
    if not metric_cols:
        raise RuntimeError("No 'task/metric' columns for the specified TASKS were found in the CSV.")
    series_by_model: Dict[str, pd.Series] = {}
    for model, g in df_ft.groupby("__model__", sort=False):
        row = g.iloc[-1]  # take last if multiple
        ser = row[metric_cols]
        ser.index = [pretty_row_label(c) for c in ser.index]
        series_by_model[model] = ser
    result_df = pd.DataFrame(series_by_model)
    # multiply acc/acc_norm by 100 with 2 d.p.
    result_df = result_df.apply(format_acc_like, axis=0)
    # sort rows by task then metric
    def row_sort_key(lbl: str):
        task, rest = lbl.split("\n", 1)
        metric = rest.strip("() ")
        order = {t: i for i, t in enumerate(TASKS)}
        return (order.get(task, 999), metric)
    result_df = result_df.reindex(sorted(result_df.index, key=row_sort_key))
    # order columns by MODEL_TOKENS
    ordered_cols = [m for m in MODEL_TOKENS if m in result_df.columns]
    other_cols = [c for c in result_df.columns if c not in ordered_cols]
    result_df = result_df[ordered_cols + other_cols]
    return result_df


# Execute
df_raw = load_wandb_export(csv_path)
results_df = build_results_table(df_raw)

# Save and display
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_CSV, encoding="utf-8")
display_dataframe_to_user("Full-tasks Results (by model)", results_df)

print(str(OUTPUT_CSV), results_df.shape)
