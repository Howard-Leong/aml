#!/usr/bin/env python3
"""
Fast reasoning-step classification on GPU.

Why this version is much faster:
- Forces CUDA and loads the model onto GPU explicitly.
- Uses a smaller default model: Qwen/Qwen2.5-1.5B-Instruct.
- Avoids text generation for labels; instead scores 6 fixed options directly.
- Uses cheap regex shortcuts for obvious Backtrack / Verification / Conclusion / Setup / Exploration steps.
- Uses larger batches by default.

Usage:
    python classify_reasoning_traces.py

Useful env vars:
    MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
    BATCH_SIZE=64
    SAVE_EVERY=100
    MAX_INPUT_TOKENS=384
    STEP_SNIPPET_CHARS=400
    CACHE_PATH="step_classifications.parquet"
    CSV_PATH="classified_traces.csv"
    LOG_FILE="classification.log"
    HF_TOKEN="hf_xxx"   # optional
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

LABELS = ["Setup", "Computation", "Verification", "Backtrack", "Exploration", "Conclusion"]

LETTER_TO_LABEL = {
    "A": "Setup",
    "B": "Computation",
    "C": "Verification",
    "D": "Backtrack",
    "E": "Exploration",
    "F": "Conclusion",
}
LETTERS = list(LETTER_TO_LABEL.keys())

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B")
CACHE_PATH = Path(os.getenv("CACHE_PATH", "step_classifications.parquet"))
CSV_PATH = Path(os.getenv("CSV_PATH", "classified_traces.csv"))
LOG_FILE = Path(os.getenv("LOG_FILE", "classification.log"))

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "384"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "100"))
STEP_SNIPPET_CHARS = int(os.getenv("STEP_SNIPPET_CHARS", "400"))

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "1").strip().lower() in {"1", "true", "yes", "y"}

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================
# Regex shortcuts
# ============================================================

BACKTRACK_RE = re.compile(
    r"\b(actually|wait[,!]?|no[,!]?|incorrect|mistake|wrong|that'?s not right|"
    r"let me reconsider|i need to reconsider|i made an error|scratch that|"
    r"hold on|this is wrong|that was wrong)\b",
    re.I,
)

VERIFICATION_RE = re.compile(
    r"\b(check|verify|verification|sanity check|double-check|confirm|"
    r"let me check|re-read|read the question again|plug (it )?back|"
    r"does this make sense|cross-check)\b",
    re.I,
)

CONCLUSION_RE = re.compile(
    r"\b(final answer|therefore|thus|hence|so the answer is|the answer is|"
    r"we conclude|conclusion|overall,? the answer)\b",
    re.I,
)

SETUP_RE = re.compile(
    r"\b(given|we are given|let(?:'s| us) define|define|denote|suppose|"
    r"the problem asks|we need to find|let x be|let y be|set )\b",
    re.I,
)

EXPLORATION_RE = re.compile(
    r"\b(maybe|perhaps|possibly|could be|might be|alternatively|another approach|"
    r"one approach|consider|let's try|we can try|case 1|case 2)\b",
    re.I,
)

# ============================================================
# Logging
# ============================================================

def setup_logging() -> logging.Logger:
    log_format = "%(asctime)s | %(levelname)8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()

# ============================================================
# Utilities
# ============================================================

def normalize_grade(x: Any) -> int:
    return 1 if str(x).strip().lower() in {"1", "true", "yes", "correct", "t", "y"} else 0


def segment_trace(trace: Any) -> List[str]:
    if not isinstance(trace, str) or not trace.strip():
        return []
    parts = re.split(r"\n{2,}", trace)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def clean_gpu_memory() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()


def get_best_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def heuristic_label(step: str) -> Optional[str]:
    if not isinstance(step, str):
        return None

    s = step.strip()
    if not s:
        return None

    if BACKTRACK_RE.search(s):
        return "Backtrack"
    if CONCLUSION_RE.search(s):
        return "Conclusion"
    if VERIFICATION_RE.search(s):
        return "Verification"
    if SETUP_RE.search(s):
        return "Setup"
    if EXPLORATION_RE.search(s):
        return "Exploration"

    return None


def build_prompt(step: str) -> str:
    snippet = str(step).strip()[:STEP_SNIPPET_CHARS]
    return (
        "Classify the reasoning step into exactly one label.\n"
        "A = Setup        (problem framing, givens, definitions)\n"
        "B = Computation  (calculations, algebra, deductions)\n"
        "C = Verification (checking work, sanity check, rereading)\n"
        "D = Backtrack    (noticing an error, revising approach)\n"
        "E = Exploration  (brainstorming, uncertainty, alternatives)\n"
        "F = Conclusion   (final answer, summary)\n\n"
        "Return only one letter: A, B, C, D, E, or F.\n\n"
        f"Reasoning step:\n{snippet}\n\n"
        "Label:"
    )


# ============================================================
# Data loading
# ============================================================

def load_and_prepare_data() -> pd.DataFrame:
    logger.info("Loading s1K dataset from HuggingFace...")

    ds = load_dataset("simplescaling/s1K-1.1")
    df = pd.DataFrame(ds["train"]).copy()
    logger.info("Loaded %d questions from s1K-1.1", len(df))

    for col in ["gemini_grade", "deepseek_grade"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_grade).astype("Int64")

    logger.info("Segmenting reasoning traces...")
    df["gemini_steps"] = df["gemini_thinking_trajectory"].apply(segment_trace)
    df["deepseek_steps"] = df["deepseek_thinking_trajectory"].apply(segment_trace)

    for m in ["gemini", "deepseek"]:
        df[f"{m}_step_count"] = df[f"{m}_steps"].str.len()
        df[f"{m}_token_count"] = (
            df[f"{m}_thinking_trajectory"].fillna("").astype(str).str.split().str.len()
        )
        df[f"{m}_backtrack_raw"] = df[f"{m}_thinking_trajectory"].apply(
            lambda t: len(BACKTRACK_RE.findall(t)) if isinstance(t, str) else 0
        )

    total_steps = int(df["gemini_step_count"].sum() + df["deepseek_step_count"].sum())
    logger.info("Total reasoning steps to classify: %s", f"{total_steps:,}")
    return df


# ============================================================
# Model loading
# ============================================================

def load_model() -> Tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This fast version requires a GPU.")

    device = torch.device("cuda:0")
    dtype = get_best_dtype()

    # Speed-friendly CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
        token=HF_TOKEN,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "right"

    logger.info("Loading model on GPU: %s", MODEL_NAME)
    last_error = None

    attn_impls: List[Optional[str]] = []
    if USE_FLASH_ATTENTION:
        attn_impls.extend(["flash_attention_2", "sdpa"])
    else:
        attn_impls.append("sdpa")
    attn_impls.append(None)

    mdl = None
    for attn_impl in attn_impls:
        try:
            kwargs = dict(
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=HF_TOKEN,
            )
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl

            mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
            mdl.to(device)
            mdl.eval()
            logger.info("Loaded model with attention implementation: %s", attn_impl or "default")
            break
        except Exception as e:
            last_error = e
            logger.warning("Model load failed with attn=%s: %s", attn_impl, e)

    if mdl is None:
        raise RuntimeError(f"Could not load model on GPU. Last error: {last_error}")

    model_device = get_input_device(mdl)
    if model_device.type != "cuda":
        raise RuntimeError(f"Model is not on GPU. Current device: {model_device}")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU confirmed: %s (%.1f GB), dtype=%s", gpu_name, gpu_mem, dtype)

    return tok, mdl


# ============================================================
# Caching
# ============================================================

def load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            cached = pd.read_parquet(path)
            logger.info("Loaded %d cached classifications from %s", len(cached), path)
            return cached
        except Exception as e:
            logger.warning("Failed to load cache: %s. Starting fresh.", e)
            return pd.DataFrame()
    logger.info("No cache found. Starting from scratch.")
    return pd.DataFrame()


def save_cache(path: Path, existing: pd.DataFrame, new_records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not new_records and existing.empty:
        return existing

    new_df = pd.DataFrame(new_records)

    if existing.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["question_id", "model"],
            keep="last",
        ).reset_index(drop=True)

    combined.to_parquet(path, index=False)
    logger.debug("Saved %d rows to cache", len(combined))
    return combined


# ============================================================
# Job queue
# ============================================================

def build_job_queue(df: pd.DataFrame, cached: pd.DataFrame) -> List[Dict[str, Any]]:
    completed = set()
    if not cached.empty and {"question_id", "model"}.issubset(cached.columns):
        completed = set(zip(cached["question_id"], cached["model"]))
        logger.info("Found %d already-classified traces in cache", len(completed))

    jobs: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        for model_key, steps_col, grade_col in [
            ("gemini", "gemini_steps", "gemini_grade"),
            ("deepseek", "deepseek_steps", "deepseek_grade"),
        ]:
            key = (int(idx), model_key)
            if key in completed:
                continue

            steps = row.get(steps_col, [])
            if not steps:
                continue

            jobs.append(
                {
                    "question_id": int(idx),
                    "model": model_key,
                    "steps": steps,
                    "correct": int(row[grade_col]) if pd.notna(row.get(grade_col)) else None,
                    "domain": row.get("source_type"),
                    "n_steps": len(steps),
                }
            )

    return jobs


# ============================================================
# Fast fixed-option scoring
# ============================================================

def score_letter_choices(prompts: Sequence[str], tok: Any, mdl: Any) -> List[str]:
    """
    Score the fixed completions ' A' ... ' F' using conditional log-probability.
    This avoids free-form generation and is much faster for tiny label sets.
    """
    device = get_input_device(mdl)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Tokenize label completions once
    label_token_ids: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids = tok(" " + letter, add_special_tokens=False)["input_ids"]
        if not ids:
            raise RuntimeError(f"Tokenizer returned empty ids for label {letter!r}")
        label_token_ids[letter] = ids

    flat_inputs: List[List[int]] = []
    flat_labels: List[List[int]] = []

    for prompt in prompts:
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

        # Keep room for label tokens
        max_prompt_len = MAX_INPUT_TOKENS - max(len(v) for v in label_token_ids.values())
        if max_prompt_len < 32:
            raise RuntimeError("MAX_INPUT_TOKENS is too small.")

        prompt_ids = prompt_ids[:max_prompt_len]

        for letter in LETTERS:
            cand_ids = label_token_ids[letter]
            input_ids = prompt_ids + cand_ids
            labels = ([-100] * len(prompt_ids)) + cand_ids
            flat_inputs.append(input_ids)
            flat_labels.append(labels)

    max_len = max(len(x) for x in flat_inputs)
    n_rows = len(flat_inputs)

    input_tensor = torch.full((n_rows, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((n_rows, max_len), dtype=torch.long, device=device)
    labels_tensor = torch.full((n_rows, max_len), -100, dtype=torch.long, device=device)

    for i, (ids, lbls) in enumerate(zip(flat_inputs, flat_labels)):
        n = len(ids)
        input_tensor[i, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
        labels_tensor[i, :n] = torch.tensor(lbls, dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = mdl(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

    # Causal LM scoring: token t predicts token t+1
    shift_logits = logits[:, :-1, :]
    shift_labels = labels_tensor[:, 1:]

    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp.masked_fill(~valid, 0.0)

    # Average label-token logprob
    seq_scores = token_logp.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    # Reshape: [num_prompts, 6 letters]
    seq_scores = seq_scores.view(len(prompts), len(LETTERS))
    best_indices = seq_scores.argmax(dim=1).tolist()

    best_letters = [LETTERS[i] for i in best_indices]
    return [LETTER_TO_LABEL[x] for x in best_letters]


def classify_steps(
    steps: Sequence[str],
    tok: Any,
    mdl: Any,
    step_bar: Optional[tqdm] = None,
) -> List[str]:
    """
    Classify one trace:
    1) regex fast-path for obvious labels
    2) fixed-option GPU scoring for the rest
    """
    results: List[Optional[str]] = [None] * len(steps)
    queued_indices: List[int] = []
    queued_prompts: List[str] = []

    heuristic_hits = 0

    for i, step in enumerate(steps):
        label = heuristic_label(step)
        if label is not None:
            results[i] = label
            heuristic_hits += 1
        else:
            queued_indices.append(i)
            queued_prompts.append(build_prompt(step))

    if step_bar is not None and heuristic_hits:
        step_bar.update(heuristic_hits)

    for start in range(0, len(queued_prompts), BATCH_SIZE):
        batch_prompts = queued_prompts[start : start + BATCH_SIZE]
        batch_indices = queued_indices[start : start + BATCH_SIZE]

        try:
            batch_labels = score_letter_choices(batch_prompts, tok, mdl)
        except Exception as e:
            logger.error("Batch option scoring failed: %s", e)
            batch_labels = ["Computation"] * len(batch_prompts)

        for idx, label in zip(batch_indices, batch_labels):
            results[idx] = label

        if step_bar is not None:
            step_bar.update(len(batch_prompts))

    return [r if r is not None else "Computation" for r in results]


# ============================================================
# Main classification loop
# ============================================================

def run_classification(df: pd.DataFrame) -> pd.DataFrame:
    cached = load_cache(CACHE_PATH)
    jobs = build_job_queue(df, cached)

    if not jobs:
        logger.info("All traces already classified. Nothing to do.")
        return cached

    total_steps = sum(job["n_steps"] for job in jobs)
    logger.info("Starting classification: %d traces, %s steps", len(jobs), f"{total_steps:,}")

    tok, mdl = load_model()

    step_bar = tqdm(
        total=total_steps,
        desc="Steps classified",
        unit="step",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    records_buffer: List[Dict[str, Any]] = []
    start_time = datetime.now()

    try:
        for k, job in enumerate(jobs, start=1):
            step_bar.set_postfix_str(
                f'{job["model"]} | q{job["question_id"]} | {job["n_steps"]} steps'
            )

            labels = classify_steps(job["steps"], tok, mdl, step_bar=step_bar)

            records_buffer.append(
                {
                    "question_id": job["question_id"],
                    "model": job["model"],
                    "labels": labels,
                    "correct": job["correct"],
                    "domain": job["domain"],
                    "n_steps": job["n_steps"],
                }
            )

            if k % SAVE_EVERY == 0:
                cached = save_cache(CACHE_PATH, cached, records_buffer)
                records_buffer.clear()
                logger.info("Checkpoint: %d/%d traces complete", k, len(jobs))
                clean_gpu_memory()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving progress...")

    finally:
        step_bar.close()

        if records_buffer:
            cached = save_cache(CACHE_PATH, cached, records_buffer)

        elapsed = datetime.now() - start_time
        logger.info("Classification complete. Time elapsed: %s", elapsed)
        logger.info("Total classified traces: %d", len(cached))

    return cached


# ============================================================
# Export
# ============================================================

def make_csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "labels" in out.columns:
        out["labels"] = out["labels"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
        )
    return out


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting FAST LLM Reasoning Trace Classification")
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Max input tokens: %d", MAX_INPUT_TOKENS)
    logger.info("Log file: %s", LOG_FILE)
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Aborting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    try:
        df = load_and_prepare_data()
        classified_df = run_classification(df)

        if not classified_df.empty:
            classified_df.to_parquet(CACHE_PATH, index=False)

            csv_df = make_csv_safe(classified_df)
            csv_df.to_csv(CSV_PATH, index=False)
            logger.info("Exported %d rows to %s", len(csv_df), CSV_PATH)

            logger.info("")
            logger.info("Summary by model:")
            for model_name in ["gemini", "deepseek"]:
                model_rows = classified_df[classified_df["model"] == model_name]
                logger.info("  %s: %d traces", model_name, len(model_rows))

        logger.info("")
        logger.info("=" * 60)
        logger.info("Classification pipeline complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()