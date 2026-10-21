"""
PRISM v1.0 — Data Preparation
Build multi-target training data from Materials Project records
(targets, missing-value masks, chemsys-based train/val/test split)
Developed by: DeviceAlchemy
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
SPDX-License-Identifier: AGPL-3.0-only
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure


REGRESSION_TARGETS = [
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "density",
    "total_magnetization_normalized_vol",
]

CLASSIFICATION_TARGETS = [
    "is_metal",       # bool
    "is_stable",      # bool
    "ordering",       # categorical: NM / FM / AFM / FiM / ...
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed line {line_no}")
    return records


def is_usable(rec: dict) -> bool:
    """Skip deprecated records and those missing structure or chemsys."""
    if rec.get("deprecated", False):
        return False
    if not rec.get("structure"):
        return False
    if not rec.get("chemsys"):
        return False
    return True


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if not is_usable(rec):
            continue
        row = {
            "material_id": rec.get("material_id"),
            "formula_pretty": rec.get("formula_pretty"),
            "chemsys": rec.get("chemsys"),
            "nelements": rec.get("nelements"),
            "structure_json": json.dumps(rec["structure"]),
        }
        for t in REGRESSION_TARGETS:
            row[t] = rec.get(t, None)
        for t in CLASSIFICATION_TARGETS:
            row[t] = rec.get(t, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    
    all_targets = REGRESSION_TARGETS + CLASSIFICATION_TARGETS
    for t in all_targets:
        df[f"{t}__mask"] = df[t].notna().astype(int)

    
    if "ordering" in df.columns:
        unknown_rows = df["ordering"] == "Unknown"
        df.loc[unknown_rows, "ordering__mask"] = 0

    return df


def chemsys_split(df: pd.DataFrame, val_frac=0.1, test_frac=0.1, seed=42):
    """Split by chemsys (e.g. 'Ba-Li-O-U') so no chemistry spans train and test."""
    rng = random.Random(seed)
    systems = df["chemsys"].unique().tolist()
    rng.shuffle(systems)

    n = len(systems)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_systems = set(systems[:n_test])
    val_systems = set(systems[n_test:n_test + n_val])
    train_systems = set(systems[n_test + n_val:])

    train_df = df[df["chemsys"].isin(train_systems)].reset_index(drop=True)
    val_df = df[df["chemsys"].isin(val_systems)].reset_index(drop=True)
    test_df = df[df["chemsys"].isin(test_systems)].reset_index(drop=True)
    return train_df, val_df, test_df


def structures_from_df(df: pd.DataFrame) -> list[Structure]:
    structs = []
    for s in df["structure_json"]:
        d = json.loads(s)
        structs.append(Structure.from_dict(d))
    return structs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    print(f"  {len(records)} raw records")

    df = build_dataframe(records)
    print(f"  {len(df)} usable records after filtering")

    for t in REGRESSION_TARGETS + CLASSIFICATION_TARGETS:
        coverage = df[f"{t}__mask"].mean() * 100
        print(f"  target coverage: {t:45s} {coverage:5.1f}%")

    train_df, val_df, test_df = chemsys_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac
    )
    print(f"  split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_df.to_parquet(args.outdir / "train.parquet")
    val_df.to_parquet(args.outdir / "val.parquet")
    test_df.to_parquet(args.outdir / "test.parquet")
    print(f"Wrote splits to {args.outdir}")


if __name__ == "__main__":
    main()
