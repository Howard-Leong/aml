#!/usr/bin/env python3
"""Two-round evaluation pipeline using one local Qwen3-8B model.

Round 1: question only
Round 2: question + extra instruction prompt

The script:
- loads a benchmark dataset from Hugging Face
- samples a configurable number of rows
- runs both prompts with the same Qwen model
- judges each answer against the reference solution with the same model
- saves a CSV checkpoint after every row
- writes a summary plot to results/analysis.png

Recommended install:
    pip install -U "transformers>=4.51.0" datasets pandas numpy matplotlib torch accelerate

Optional (for lower VRAM):
    pip install bitsandbytes
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_DATASET = "simplescaling/s1K-1.1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_RESULTS_CSV = DEFAULT_RESULTS_DIR / "benchmark_results.csv"
DEFAULT_ANALYSIS_PNG = DEFAULT_RESULTS_DIR / "analysis.png"

# Round 2 prompt stays close to the original code you shared.
ROUND2_PROMPT_SUFFIX = "When you're thinking, conduct verification by checking your work, sanity-checking your steps, and re-reading the question to ensure you understand it correctly. Then, generate your final response after this verification step."

# Judge prompt: kept on the same model family for a fully local pipeline.
JUDGE_PROMPT = """
You are an Al assistant for grading a science problem. The user will provide you with the question itself, an attempt made by a student and the correct answer to the problem. Your job is to judge whether the attempt is correct by comparing it with the correct answer. If the expected solution concludes with a number or choice, there should be no ambiguity. If the expected solution involves going through the entire reasoning process, you should judge the attempt based on whether the reasoning process is correct with correct answer if helpful.
The user will provide the attempt and the correct answer in the following format:
# Problem
{problem}
## Attempt
{attempt}
## Correct answer
{solution}
Explain your reasoning, and endyour response on a new line with only "Yes"or "No" (without quotes).
"""

# Column aliases for datasets that do not use the exact names we want.
QUESTION_ALIASES = ["question", "problem", "prompt", "input", "query"]
SOLUTION_ALIASES = ["solution", "answer", "output", "correct_answer", "reference"]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("qwen-two-round-eval")


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Remove common Qwen thinking wrappers and trim whitespace."""
    if text is None:
        return ""

    cleaned = text.strip()

    # Remove explicit <think> blocks if they appear in decoded output.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*</?think>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    return cleaned


def last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_yes_no(text: str) -> str:
    """Extract a Yes/No verdict from the judge response."""
    last = last_nonempty_line(text).lower()
    if last == "yes" or last.endswith(" yes"):
        return "Yes"
    if last == "no" or last.endswith(" no"):
        return "No"

    # Fallback: search the whole string, preferring the last occurrence.
    lower = text.lower()
    yes_pos = lower.rfind("yes")
    no_pos = lower.rfind("no")
    if yes_pos == -1 and no_pos == -1:
        return "Unknown"
    return "Yes" if yes_pos > no_pos else "No"


def normalize_grade(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.capitalize()
        .map(lambda x: x if x in {"Yes", "No"} else "Unknown")
    )


def accuracy(series: pd.Series) -> float:
    valid = series[series.isin(["Yes", "No"])]
    if len(valid) == 0:
        return 0.0
    return float((valid == "Yes").mean() * 100.0)


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def load_benchmark_dataset(dataset_name: str, sample_size: int, seed: int) -> pd.DataFrame:
    log.info("Loading dataset: %s", dataset_name)
    ds = load_dataset(dataset_name)

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    df.columns = [c.lower().strip() for c in df.columns]
    log.info("Dataset split=%s | rows=%d | columns=%s", split_name, len(df), list(df.columns))

    if "question" not in df.columns:
        for alias in QUESTION_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: "question"})
                log.info("Renamed column '%s' -> 'question'", alias)
                break

    if "solution" not in df.columns:
        for alias in SOLUTION_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: "solution"})
                log.info("Renamed column '%s' -> 'solution'", alias)
                break

    missing = [c for c in ["question", "solution"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find required columns {missing} in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    sample_size = min(sample_size, len(df))
    df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    log.info("Sampled %d rows", len(df))
    return df


# -----------------------------------------------------------------------------
# Model loading / generation
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, hf_token: Optional[str], device_map: str = "auto"):
    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype="auto",
        device_map=device_map,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    log.info("Model ready")
    return model, tokenizer


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    user_prompt: str,
    *,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """Return (raw_generation, cleaned_answer)."""
    messages = [{"role": "user", "content": user_prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer([chat_text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Remove None values to keep generate() happy.
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    output_ids = model.generate(**gen_kwargs)
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    cleaned = clean_text(raw)
    return raw.strip(), cleaned


# -----------------------------------------------------------------------------
# Judge
# -----------------------------------------------------------------------------

def judge_answer(
    model,
    tokenizer,
    question: str,
    attempt: str,
    solution: str,
    *,
    max_new_tokens: int,
) -> Tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question, attempt=attempt, solution=solution)
    raw, cleaned = generate_answer(
        model,
        tokenizer,
        prompt,
        enable_thinking=False,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    verdict = extract_yes_no(cleaned if cleaned else raw)
    return raw, verdict


# -----------------------------------------------------------------------------
# Checkpoint handling
# -----------------------------------------------------------------------------

def load_checkpoint(results_csv: Path) -> Optional[pd.DataFrame]:
    if results_csv.exists():
        log.info("Loading checkpoint: %s", results_csv)
        return pd.read_csv(results_csv)
    return None


def save_checkpoint(df: pd.DataFrame, results_csv: Path, results_dir: Path) -> None:
    ensure_dir(results_dir)
    df.to_csv(results_csv, index=False)


# -----------------------------------------------------------------------------
# Analysis / plotting
# -----------------------------------------------------------------------------

def analyze_and_plot(df: pd.DataFrame, analysis_png: Path) -> Dict[str, Any]:
    round1 = normalize_grade(df["round1_judge"])
    round2 = normalize_grade(df["round2_judge"])

    acc1 = accuracy(round1)
    acc2 = accuracy(round2)

    valid_mask = round1.isin(["Yes", "No"]) & round2.isin(["Yes", "No"] )
    both_valid = df[valid_mask].copy()
    agree = int((round1[valid_mask] == round2[valid_mask]).sum())

    r1_yes_r2_yes = int(((round1 == "Yes") & (round2 == "Yes")).sum())
    r1_yes_r2_no = int(((round1 == "Yes") & (round2 == "No")).sum())
    r1_no_r2_yes = int(((round1 == "No") & (round2 == "Yes")).sum())
    r1_no_r2_no = int(((round1 == "No") & (round2 == "No")).sum())

    print("\n" + "═" * 60)
    print("  BENCHMARK COMPARISON SUMMARY")
    print("═" * 60)
    print(f"  Round 1 accuracy (question only)     : {acc1:.1f}%")
    print(f"  Round 2 accuracy (extra instruction)  : {acc2:.1f}%")
    print(f"  Delta                                : {acc2 - acc1:+.1f}%")
    print(f"  Agreement between rounds             : {agree}/{len(both_valid)} ({100 * agree / max(len(both_valid), 1):.1f}%)")
    print(f"  Both correct  (Yes/Yes)              : {r1_yes_r2_yes}")
    print(f"  R1 right, R2 wrong  (Yes/No)         : {r1_yes_r2_no}")
    print(f"  R1 wrong, R2 right  (No/Yes)         : {r1_no_r2_yes}")
    print(f"  Both wrong   (No/No)                 : {r1_no_r2_no}")
    print("═" * 60 + "\n")

    # Plot
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    colors = {
        "yes": "#4ade80",
        "no": "#f87171",
        "unknown": "#94a3b8",
        "r1": "#60a5fa",
        "r2": "#f472b6",
    }

    def style_ax(ax, title: str):
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    # Accuracy bars
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["Round 1", "Round 2"]
    values = [acc1, acc2]
    bars = ax1.bar(labels, values, color=[colors["r1"], colors["r2"]], width=0.55, zorder=3)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Accuracy (%)", color="white", fontsize=9)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    ax1.axhline(y=50, color="#6b7280", linestyle="--", linewidth=0.8, zorder=2)
    style_ax(ax1, "Accuracy Comparison")

    # Round 1 pie
    ax2 = fig.add_subplot(gs[0, 1])
    vc1 = round1.value_counts()
    wedge_colors1 = [colors.get(str(k).lower(), colors["unknown"]) for k in vc1.index]
    _, _, autotexts1 = ax2.pie(
        vc1.values,
        labels=vc1.index,
        autopct="%1.1f%%",
        colors=wedge_colors1,
        startangle=90,
        textprops={"color": "white", "fontsize": 9},
    )
    for at in autotexts1:
        at.set_color("white")
    style_ax(ax2, "Round 1 Grade Distribution")

    # Round 2 pie
    ax3 = fig.add_subplot(gs[0, 2])
    vc2 = round2.value_counts()
    wedge_colors2 = [colors.get(str(k).lower(), colors["unknown"]) for k in vc2.index]
    _, _, autotexts2 = ax3.pie(
        vc2.values,
        labels=vc2.index,
        autopct="%1.1f%%",
        colors=wedge_colors2,
        startangle=90,
        textprops={"color": "white", "fontsize": 9},
    )
    for at in autotexts2:
        at.set_color("white")
    style_ax(ax3, "Round 2 Grade Distribution")

    # Agreement matrix
    ax4 = fig.add_subplot(gs[1, 0:2])
    matrix = np.array(
        [
            [r1_yes_r2_yes, r1_yes_r2_no],
            [r1_no_r2_yes, r1_no_r2_no],
        ]
    )
    ax4.imshow(matrix, cmap="YlGn", aspect="auto")
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["R2: Yes", "R2: No"], color="white")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["R1: Yes", "R1: No"], color="white")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
    style_ax(ax4, "Round 1 vs Round 2 Agreement Matrix")

    # Breakdown bar chart
    ax5 = fig.add_subplot(gs[1, 2])
    breakdown = {
        "R1✓ R2✓": r1_yes_r2_yes,
        "R1✓ R2✗": r1_yes_r2_no,
        "R1✗ R2✓": r1_no_r2_yes,
        "R1✗ R2✗": r1_no_r2_no,
    }
    bars5 = ax5.bar(list(breakdown.keys()), list(breakdown.values()), color=["#4ade80", "#f87171", "#60a5fa", "#fbbf24"], zorder=3)
    ax5.set_ylabel("Count", color="white", fontsize=9)
    for bar in bars5:
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
        )
    ax5.tick_params(axis="x", labelsize=8)
    style_ax(ax5, "Outcome Breakdown")

    fig.suptitle("LLM Benchmark: Round 1 vs Round 2", color="white", fontsize=15, fontweight="bold", y=0.98)
    ensure_dir(analysis_png.parent)
    plt.savefig(analysis_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "round1_accuracy": acc1,
        "round2_accuracy": acc2,
        "delta": acc2 - acc1,
        "agreement_count": agree,
        "agreement_rate": 100.0 * agree / max(len(df), 1),
        "both_valid_count": int(len(both_valid)),
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the output dataframe has every expected column."""
    expected_cols = [
        "round1_prompt",
        "round1_raw",
        "round1_final",
        "round1_judge_raw",
        "round1_judge",
        "round2_prompt",
        "round2_raw",
        "round2_final",
        "round2_judge_raw",
        "round2_judge",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dir(args.results_dir)

    hf_token = args.hf_token or None
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    df = load_benchmark_dataset(args.dataset, args.sample_size, args.seed)
    checkpoint = load_checkpoint(args.results_csv)

    if checkpoint is not None:
        # Prefer checkpoint content when present; otherwise merge with the fresh sample.
        checkpoint = prepare_dataframe(checkpoint)
        if len(checkpoint) >= len(df):
            df = checkpoint.head(len(df)).copy()
        else:
            # Merge checkpoint into the sampled rows by index.
            df = prepare_dataframe(df)
            for col in checkpoint.columns:
                if col not in df.columns:
                    df[col] = ""
            for idx in checkpoint.index:
                if idx < len(df):
                    for col in checkpoint.columns:
                        df.at[idx, col] = checkpoint.at[idx, col]
        log.info("Resuming from checkpoint")
    else:
        df = prepare_dataframe(df)

    model, tokenizer = load_model_and_tokenizer(args.model_id, hf_token, device_map=args.device_map)

    for idx in range(len(df)):
        question = str(df.at[idx, "question"])
        solution = str(df.at[idx, "solution"])

        round1_prompt = question
        round2_prompt = f"{question}\n{ROUND2_PROMPT_SUFFIX}"

        df.at[idx, "round1_prompt"] = round1_prompt
        df.at[idx, "round2_prompt"] = round2_prompt

        # Only generate missing rows so checkpoint resume works cleanly.
        if not str(df.at[idx, "round1_final"]).strip():
            raw1, final1 = generate_answer(
                model,
                tokenizer,
                round1_prompt,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            df.at[idx, "round1_raw"] = raw1
            df.at[idx, "round1_final"] = final1

        if not str(df.at[idx, "round2_final"]).strip():
            raw2, final2 = generate_answer(
                model,
                tokenizer,
                round2_prompt,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            df.at[idx, "round2_raw"] = raw2
            df.at[idx, "round2_final"] = final2

        if not str(df.at[idx, "round1_judge"]).strip():
            judge_raw1, judge1 = judge_answer(
                model,
                tokenizer,
                question,
                str(df.at[idx, "round1_raw"]),
                solution,
                max_new_tokens=args.judge_max_new_tokens,
            )
            df.at[idx, "round1_judge_raw"] = judge_raw1
            df.at[idx, "round1_judge"] = judge1

        if not str(df.at[idx, "round2_judge"]).strip():
            judge_raw2, judge2 = judge_answer(
                model,
                tokenizer,
                question,
                str(df.at[idx, "round2_raw"]),
                solution,
                max_new_tokens=args.judge_max_new_tokens,
            )
            df.at[idx, "round2_judge_raw"] = judge_raw2
            df.at[idx, "round2_judge"] = judge2

        save_checkpoint(df, args.results_csv, args.results_dir)
        log.info("Finished row %d/%d", idx + 1, len(df))

    stats = analyze_and_plot(df, args.analysis_png)
    save_checkpoint(df, args.results_csv, args.results_dir)

    log.info("Done. Saved CSV to %s", args.results_csv)
    log.info("Saved plot to %s", args.analysis_png)
    log.info("Summary: %s", stats)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-round Qwen3 evaluation pipeline")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Output directory")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV, help="CSV output path")
    parser.add_argument("--analysis-png", type=Path, default=DEFAULT_ANALYSIS_PNG, help="Plot output path")
    parser.add_argument("--max-new-tokens", type=int, default=100000, help="Generation limit for answers")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256, help="Generation limit for judge responses")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for both rounds")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for both rounds")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True, help="Use Qwen3 thinking mode for both rounds")
    parser.add_argument("--device-map", default="auto", help='Transformers device_map, e.g. "auto"')
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="Optional Hugging Face token")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
