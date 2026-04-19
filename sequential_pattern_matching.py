"""
Section 10: Sequential Pattern Mining (PrefixSpan)
===================================================
Parses reasoning traces, collapses repeated labels, and mines frequent
sequential patterns using a PrefixSpan-style algorithm.

This version is designed to handle traces stored in either:
- comma-separated CSV, where labels is a NumPy-style string repr
- whitespace-separated rows, where labels is the bracketed NumPy-style repr

Examples of labels values:
- ["Setup", "Computation", "Verification"]
- ['Setup' 'Computation' 'Verification']
- ['Setup' 'Computation' 'Setup' 'Setup' 'Backtrack' ...]   ← truncated (handled)

Expected input
--------------
classified_df : pd.DataFrame
    Must contain at minimum:
        - "labels"  : list[str] or serialised sequence string
        - "model"   : str        — "gemini" or "deepseek"
        - "correct" : int        — 1 = correct, 0 = incorrect

Usage
-----
Standalone:

    python section_10_sequential_pattern_mining_prefixspan.py

Or import:

    from section_10_sequential_pattern_mining_prefixspan import run_pattern_mining
    pattern_results = run_pattern_mining(classified_df)
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LABELS = [
    "Setup",
    "Computation",
    "Verification",
    "Backtrack",
    "Exploration",
    "Conclusion",
]
LABEL_SET = set(LABELS)
EPS = 1e-9

# -----------------------------------------------------------------------------
# Parsing and validation
# -----------------------------------------------------------------------------

def parse_labels(x) -> List[str]:
    """Convert a serialised trace into a clean Python list[str].

    Handles:
    - Already-parsed list/tuple
    - Proper Python literals: ["a", "b"]
    - NumPy-style repr (no commas): ['Setup' 'Computation']
    - NumPy-truncated repr with '...': ['Setup' 'Computation' ... 'Conclusion']
      → '...' tokens are silently dropped; partial data is returned
    - Comma-separated fallback
    - Whitespace-separated fallback
    """
    if isinstance(x, (list, tuple)):
        return [str(v).strip().strip('"').strip("'") for v in x if str(v).strip()]

    if pd.isna(x) or not isinstance(x, str):
        return []

    s = x.strip()
    if not s:
        return []

    # --- NumPy-style repr (with or without truncation '...') -----------------
    # Detected by: starts with '[' and contains single-quoted tokens.
    # We use findall so that '...' between tokens is automatically skipped.
    if s.startswith("[") and "'" in s:
        tokens = re.findall(r"'([^']*)'", s, flags=re.DOTALL)
        if tokens:
            return [t.strip() for t in tokens if t.strip()]

    # --- Proper Python list/tuple literal (no truncation) --------------------
    # Only attempt after the NumPy path, because ast.literal_eval will raise
    # SyntaxError on truncated strings containing '...'.
    if s.startswith(("[", "(", "{")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                out = [str(v).strip().strip('"').strip("'") for v in parsed]
                return [v for v in out if v]
        except Exception:
            pass

    # --- Comma-separated fallback --------------------------------------------
    if "," in s:
        parts = [t.strip().strip('"').strip("'") for t in s.split(",")]
        parts = [p for p in parts if p]
        if len(parts) > 1:
            return parts

    # --- Whitespace-separated fallback (last resort) -------------------------
    parts = [t.strip().strip('"').strip("'") for t in s.split()]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        return parts

    return [s]


def collapse_runs(seq: Sequence[str]) -> List[str]:
    """Remove consecutive duplicates so the sequence keeps only transitions."""
    seq = [str(x).strip() for x in seq if str(x).strip()]
    if not seq:
        return []

    out = [seq[0]]
    for item in seq[1:]:
        if item != out[-1]:
            out.append(item)
    return out


def normalize_sequence(x) -> List[str]:
    """Parse and collapse a sequence into a clean list[str]."""
    return collapse_runs(parse_labels(x))


def validate_sequences(df: pd.DataFrame, verbose: bool = True) -> None:
    """Print quick diagnostics about label sequences."""
    if "labels" not in df.columns:
        raise ValueError('DataFrame must contain a "labels" column.')

    bad_type = df["labels"].apply(lambda x: not isinstance(x, list))
    if bad_type.any() and verbose:
        print(f"⚠️  {bad_type.sum()} rows are not parsed as lists.")

    lens = df["labels"].apply(len)
    if verbose:
        print(
            f"Parsed labels — rows: {len(df)}, "
            f"median seq len: {int(lens.median()) if len(lens) else 0}, "
            f"empty seqs: {(lens == 0).sum()}, "
            f"single-element seqs: {(lens == 1).sum()} "
            f"(single-element seqs are valid but won't yield 2-step patterns)"
        )

    # BUG FIX: explode() on an all-empty column raises; guard with a check
    exploded = df["labels"].explode().dropna().astype(str)
    if exploded.empty:
        if verbose:
            print("⚠️  No label values found after exploding — all sequences empty?")
        return

    observed = sorted(exploded.unique().tolist())
    unknown = [x for x in observed if x not in LABEL_SET]
    if verbose and unknown:
        preview = unknown[:20]
        suffix = " ..." if len(unknown) > 20 else ""
        print(f"⚠️  Unknown labels detected ({len(unknown)}): {preview}{suffix}")


# -----------------------------------------------------------------------------
# File loader
# -----------------------------------------------------------------------------

def read_classified_traces(path: str) -> pd.DataFrame:
    """Read classified traces from either CSV or whitespace-separated rows.

    Supported formats:

    1) CSV:
       question_id,model,labels,correct,domain,n_steps
       0,gemini,"['Setup' 'Computation' 'Verification']",1,qq8933/AIME_1983_2024,53

    2) Whitespace-separated:
       0 gemini ['Setup' 'Computation' 'Verification'] 1 qq8933/AIME_1983_2024 53
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f if line.strip()]

    if not raw:
        raise ValueError(f"Empty input file: {path}")

    header = raw[0].strip()

    # Case 1: ordinary CSV
    if "," in header:
        # BUG FIX: previous implementation double-appended lines.
        # We rebuild a single clean string by merging lines that belong to the
        # same CSV record (i.e. lines where the '[...]' bracket is not yet closed).
        merged: List[str] = []
        buffer = ""
        bracket_balance = 0

        for line in raw:
            if bracket_balance <= 0 and buffer:
                # Previous record is complete — flush it before starting new one
                merged.append(buffer)
                buffer = ""

            if not buffer:
                buffer = line.strip()
            else:
                buffer += " " + line.strip()

            bracket_balance += line.count("[") - line.count("]")

        # Flush the last buffered record
        if buffer:
            merged.append(buffer)

        df = pd.read_csv(io.StringIO("\n".join(merged)))
        return df

    # Case 2: whitespace-separated rows
    pattern = re.compile(
        r"^(\S+)\s+(\S+)\s+(\[.*\])\s+(\S+)\s+(\S+)\s+(\S+)$"
    )
    columns = ["question_id", "model", "labels", "correct", "domain", "n_steps"]

    data_lines = (
        raw[1:]
        if "question_id" in header.lower() and "labels" in header.lower()
        else raw
    )

    rows = []
    for line in data_lines:
        m = pattern.match(line.strip())
        if not m:
            raise ValueError(
                "Could not parse whitespace-separated row.\n"
                f"Offending line:\n{line}"
            )
        rows.append(dict(zip(columns, m.groups())))

    df = pd.DataFrame(rows)

    for col in ["question_id", "correct", "n_steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# -----------------------------------------------------------------------------
# PrefixSpan implementation
# -----------------------------------------------------------------------------

Pattern = Tuple[str, ...]


@dataclass(frozen=True)
class PatternRecord:
    pattern: Pattern
    support: float
    support_count: int


def _project_sequence(seq: Sequence[str], pattern: Pattern) -> Optional[int]:
    """Return the first index after the last element of pattern in seq.

    Returns None if pattern is not a subsequence of seq.
    Returns 0 for an empty pattern (every sequence matches).
    """
    if not pattern:
        return 0

    start = 0
    for token in pattern:
        found = False
        for idx in range(start, len(seq)):
            if seq[idx] == token:
                start = idx + 1
                found = True
                break
        if not found:
            return None
    return start


def _candidate_items_from_projected_db(
    sequences: Sequence[Sequence[str]],
    pattern: Pattern,
    max_gap: Optional[int] = None,
) -> Dict[str, int]:
    """Count distinct sequence-support for possible next tokens."""
    counts: Dict[str, int] = defaultdict(int)

    for seq in sequences:
        start = _project_sequence(seq, pattern)

        # BUG FIX: explicitly skip sequences where the pattern matches at the
        # very end — there are no items left to extend with.
        if start is None or start >= len(seq):
            continue

        end = len(seq) if max_gap is None else min(len(seq), start + max_gap)
        seen: set = set()
        for token in seq[start:end]:
            if token not in seen:
                counts[token] += 1
                seen.add(token)

    return counts


def prefixspan_mine(
    sequences: Sequence[Sequence[str]],
    min_support: float = 0.06,
    max_pattern_len: int = 5,
    max_gap: Optional[int] = None,
) -> pd.DataFrame:
    """Mine frequent sequential patterns using a PrefixSpan-style recursion."""
    sequences = [tuple(seq) for seq in sequences if seq]
    if not sequences:
        return pd.DataFrame(columns=["support", "support_count", "pattern"])

    n_sequences = len(sequences)
    min_count = max(1, int(np.ceil(min_support * n_sequences)))

    results: List[PatternRecord] = []

    def recurse(prefix: Pattern) -> None:
        if len(prefix) >= max_pattern_len:
            return

        counts = _candidate_items_from_projected_db(sequences, prefix, max_gap=max_gap)
        if not counts:
            return

        for token in sorted(counts.keys()):
            support_count = counts[token]
            if support_count < min_count:
                continue

            new_pattern = prefix + (token,)
            support = support_count / n_sequences
            results.append(
                PatternRecord(
                    pattern=new_pattern,
                    support=support,
                    support_count=support_count,
                )
            )
            recurse(new_pattern)

    recurse(())

    if not results:
        return pd.DataFrame(columns=["support", "support_count", "pattern"])

    df = pd.DataFrame(
        {
            "support": [r.support for r in results],
            "support_count": [r.support_count for r in results],
            "pattern": [" → ".join(r.pattern) for r in results],
        }
    )

    df = df.sort_values(
        by=["support", "support_count", "pattern"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Splits and mining orchestration
# -----------------------------------------------------------------------------

def build_splits(classified_df: pd.DataFrame) -> Dict[tuple, List[List[str]]]:
    """Partition sequences by (model, correct)."""
    required = {"labels", "model", "correct"}
    missing = required - set(classified_df.columns)
    if missing:
        raise ValueError(f"classified_df is missing columns: {sorted(missing)}")

    def get_seqs(model: str, correct: int) -> List[List[str]]:
        mask = (classified_df["model"] == model) & (classified_df["correct"] == correct)
        seqs = classified_df.loc[mask, "labels"].tolist()
        return [s for s in seqs if isinstance(s, list)]

    return {
        ("gemini", 1): get_seqs("gemini", 1),
        ("gemini", 0): get_seqs("gemini", 0),
        ("deepseek", 1): get_seqs("deepseek", 1),
        ("deepseek", 0): get_seqs("deepseek", 0),
    }


def run_pattern_mining(
    classified_df: pd.DataFrame,
    min_support: float = 0.06,
    max_pattern_len: int = 5,
    max_gap: Optional[int] = None,
) -> Dict[str, pd.Series]:
    """Mine frequent sequential patterns for each (model × correctness) split."""
    df = classified_df.copy()
    df["labels"] = df["labels"].apply(normalize_sequence)

    validate_sequences(df, verbose=True)
    splits = build_splits(df)

    all_seqs = [seq for seqs in splits.values() for seq in seqs]
    seq_lens = [len(s) for s in all_seqs if s]
    if seq_lens:
        frac_short = sum(l < 2 for l in seq_lens) / len(seq_lens)
        print(
            f"  Sequence length stats — min={min(seq_lens)}, "
            f"median={int(np.median(seq_lens))}, "
            f"max={max(seq_lens)}, "
            f"frac_shorter_than_2={frac_short:.2%}"
        )
    else:
        print("  WARNING: all label sequences are empty after parsing!")

    # BUG FIX: the original dedup used a list-comprehension that iterated the
    # list it was filtering mid-construction, so it never actually deduplicated.
    # dict.fromkeys preserves insertion order and correctly removes duplicates.
    raw_fallbacks = [min_support, 0.03, 0.01, 0.005]
    fallback_supports = list(dict.fromkeys(raw_fallbacks))

    print("Mining sequential patterns …")
    pattern_results: Dict[str, pd.Series] = {}

    for (mdl, correct), seqs in splits.items():
        key = f"{mdl}_{'C' if correct else 'I'}"

        result = pd.DataFrame()
        used_support = None
        for sup in fallback_supports:
            result = prefixspan_mine(
                seqs,
                min_support=sup,
                max_pattern_len=max_pattern_len,
                max_gap=max_gap,
            )
            if not result.empty:
                used_support = sup
                break

        if result.empty:
            pattern_results[key] = pd.Series(dtype=float)
            eligible = sum(len(s) >= 2 for s in seqs)
            print(
                f"  {key}: 0 patterns found "
                f"(split size={len(seqs)}, eligible seqs={eligible})"
            )
        else:
            pattern_results[key] = result.set_index("pattern")["support"]
            if used_support is not None and used_support != min_support:
                print(
                    f"  {key}: min_support auto-lowered to {used_support} → {len(result)} patterns"
                )
            else:
                print(f"  {key}: {len(result)} patterns found")

    return pattern_results


# -----------------------------------------------------------------------------
# Enrichment analysis
# -----------------------------------------------------------------------------

def enrichment_analysis(
    pattern_results: Dict[str, pd.Series],
    top_k: int = 6,
) -> None:
    """Print patterns enriched in correct vs incorrect traces and vice versa."""
    print("=== Patterns enriched in CORRECT vs INCORRECT traces ===\n")
    for mdl in ["gemini", "deepseek"]:
        c = pattern_results.get(f"{mdl}_C", pd.Series(dtype=float))
        i = pattern_results.get(f"{mdl}_I", pd.Series(dtype=float))

        if c.empty or i.empty:
            print(f"--- {mdl.upper()}: insufficient data to compute enrichment ---\n")
            continue

        common = c.index.intersection(i.index)
        if common.empty:
            print(f"--- {mdl.upper()}: no overlapping patterns found ---\n")
            continue

        ratio = (c[common] / (i[common] + EPS)).rename("ratio")
        ratio_df = pd.concat(
            [c[common].rename("correct"), i[common].rename("incorrect"), ratio],
            axis=1,
        )

        print(f"--- {mdl.upper()}: Enriched in CORRECT ---")
        print(
            ratio_df.nlargest(top_k, "ratio")[["correct", "incorrect", "ratio"]]
            .round(3)
            .to_string()
        )
        print(f"\n--- {mdl.upper()}: Enriched in INCORRECT ---")
        print(
            ratio_df.nsmallest(top_k, "ratio")[["correct", "incorrect", "ratio"]]
            .round(3)
            .to_string(),
            "\n",
        )


# -----------------------------------------------------------------------------
# Support comparison table
# -----------------------------------------------------------------------------

def support_comparison_table(pattern_results: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build a wide table of supports and ratios across all splits."""
    frames = {k: v for k, v in pattern_results.items() if not v.empty}
    if not frames:
        return pd.DataFrame(columns=list(pattern_results.keys()))

    all_patterns: set = set()
    for s in frames.values():
        all_patterns.update(s.index.tolist())

    rows = []
    for pat in sorted(all_patterns):
        row = {"pattern": pat}
        for key, series in pattern_results.items():
            # BUG FIX: .at[] raises KeyError if the index has duplicate entries.
            # .get() is safe and returns the default when the key is absent.
            row[key] = float(series.get(pat, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("pattern")
    for mdl in ["gemini", "deepseek"]:
        c_col = f"{mdl}_C"
        i_col = f"{mdl}_I"
        if c_col in df.columns and i_col in df.columns:
            df[f"{mdl}_ratio"] = df[c_col] / (df[i_col] + EPS)

    return df.round(4)


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def plot_top_patterns(
    pattern_results: Dict[str, pd.Series],
    top_k: int = 10,
    figsize: tuple = (14, 8),
) -> None:
    """Bar charts of top-k most frequent patterns for each split."""
    keys = [
        k for k in ["gemini_C", "gemini_I", "deepseek_C", "deepseek_I"]
        if k in pattern_results
    ]
    n_panels = len(keys)
    if n_panels == 0:
        print("No pattern results to plot.")
        return

    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    titles = {
        "gemini_C": "Gemini — Correct",
        "gemini_I": "Gemini — Incorrect",
        "deepseek_C": "DeepSeek — Correct",
        "deepseek_I": "DeepSeek — Incorrect",
    }

    all_axes = axes.flat
    for idx, key in enumerate(keys):
        ax = all_axes[idx]
        series = pattern_results[key]
        if series.empty:
            ax.set_title(f"{titles.get(key, key)} (no patterns)")
            ax.axis("off")
            continue

        top = series.head(top_k)
        ax.barh(top.index[::-1], top.values[::-1], edgecolor="white")
        ax.set_title(titles.get(key, key))
        ax.set_xlabel("Support")
        ax.tick_params(axis="y", labelsize=8)

    # BUG FIX: hide any unused axes without risking an index out-of-range
    for idx in range(n_panels, n_rows * n_cols):
        all_axes[idx].axis("off")

    plt.suptitle("Top Frequent Sequential Patterns by Model × Correctness", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_enrichment_bars(
    pattern_results: Dict[str, pd.Series],
    top_k: int = 8,
    figsize: tuple = (14, 6),
) -> None:
    """Plot log2 enrichment ratios for patterns shared by correct/incorrect splits."""
    models = [
        m for m in ["gemini", "deepseek"]
        if f"{m}_C" in pattern_results and f"{m}_I" in pattern_results
    ]
    if not models:
        print("No paired (correct + incorrect) pattern data to plot.")
        return

    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)

    for ax, mdl in zip(axes.flat, models):
        c = pattern_results[f"{mdl}_C"]
        i = pattern_results[f"{mdl}_I"]
        common = c.index.intersection(i.index)
        if common.empty:
            ax.set_title(f"{mdl.upper()} — no overlap")
            ax.axis("off")
            continue

        ratio = c[common] / (i[common] + EPS)
        top_correct = ratio.nlargest(top_k)
        top_incorrect = ratio.nsmallest(top_k)
        combined = pd.concat([top_correct, top_incorrect]).drop_duplicates().sort_values()

        colors = ["#2ecc71" if v >= 1 else "#e74c3c" for v in combined.values]
        ax.barh(combined.index, np.log2(combined.values + EPS), color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{mdl.upper()}: Pattern Enrichment\n(green=correct, red=incorrect)")
        ax.set_xlabel("log₂(correct support / incorrect support)")
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("Pattern Enrichment: Correct vs Incorrect Traces", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Standalone entry point."""
    debug = "--debug" in sys.argv
    csv_path = os.environ.get("CLASSIFIED_CSV", "classified_traces.csv")
    print(f"Loading classified traces from: {csv_path}")

    classified_df = read_classified_traces(csv_path)
    if "labels" not in classified_df.columns:
        raise ValueError('Input file must contain a "labels" column.')

    classified_df["labels"] = classified_df["labels"].apply(normalize_sequence)

    validate_sequences(classified_df, verbose=True)

    if debug:
        print("\n[DEBUG] Parsed labels sample (first 3 rows):")
        for i, v in classified_df["labels"].head(3).items():
            print(f"  row {i}: {v[:12]} ... ({len(v)} labels)")

    pattern_results = run_pattern_mining(
        classified_df,
        min_support=0.06,
        max_pattern_len=5,
        max_gap=None,
    )

    print()
    enrichment_analysis(pattern_results, top_k=6)

    print("=== Full support comparison table ===")
    tbl = support_comparison_table(pattern_results)
    if tbl.empty:
        print("  (no patterns found — see diagnostics above)")
    else:
        print(tbl.head(20).to_string())
        print(f"\n({len(tbl)} total patterns across all splits)\n")

    if not all(s.empty for s in pattern_results.values()):
        plot_top_patterns(pattern_results, top_k=10)
        plot_enrichment_bars(pattern_results, top_k=8)
    else:
        print("Skipping plots — no patterns to display.")


if __name__ == "__main__":
    main()