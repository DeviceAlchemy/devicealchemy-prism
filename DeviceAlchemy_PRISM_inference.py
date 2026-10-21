"""
PRISM v1.0 — Inference
Developed by: DeviceAlchemy
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
SPDX-License-Identifier: AGPL-3.0-only
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure, Composition

import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader
from matminer.featurizers.composition import ElementProperty

from DeviceAlchemy_PRISM_Data_Preparation import REGRESSION_TARGETS, CLASSIFICATION_TARGETS
from DeviceAlchemy_PRISM_training import (
    MultiTaskHeads,
    multitask_collate,
    unpack_labels,
    REG_KEYS,
    CLS_KEYS,
    ORDERING_CLASSES,
    DEVICE,
)


DEFAULT_THRESHOLDS = {
    "max_energy_above_hull": 0.05,
    "require_is_stable": False,   
    "band_gap_range": None,       
}


class StackScreener:
    def __init__(self, gnn_checkpoint: str, baseline_dir: str = None,
                 pretrained: str = "M3GNet-Eform-MP-2019.4.1"):
        self.encoder = matgl.load_model(pretrained).to(DEVICE)
        self.converter = Structure2Graph(
            element_types=self.encoder.element_types, cutoff=self.encoder.cutoff
        )
        self.heads = MultiTaskHeads(n_reg=len(REG_KEYS)).to(DEVICE)
        
        dummy = Structure(
            lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            species=["Fe"], coords=[[0, 0, 0]],
        )
        self._prime(dummy)

        ckpt = torch.load(gnn_checkpoint, map_location=DEVICE)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.heads.load_state_dict(ckpt["heads"])
        self.encoder.eval()
        self.heads.eval()

        stats_path = Path(gnn_checkpoint).parent / "target_stats.npy"
        self.stats = np.load(stats_path, allow_pickle=True).item()

        self.baseline_density_model = None
        self.magpie = None
        if baseline_dir:
            density_path = Path(baseline_dir) / "density_xgb.json"
            if density_path.exists():
                import xgboost as xgb
                self.baseline_density_model = xgb.XGBRegressor()
                self.baseline_density_model.load_model(str(density_path))
                self.magpie = ElementProperty.from_preset("magpie")
            else:
                print(f"[warn] {density_path} not found — density will come from the GNN instead")

    def _prime(self, structure):
        g, lat, state_attr = self.converter.get_graph(structure)
        g.pos = g.frac_coords @ lat[0]
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        state_attr = torch.as_tensor(state_attr, dtype=torch.float32)
        with torch.no_grad():
            _ = self.encoder(g.to(DEVICE), state_attr.to(DEVICE).unsqueeze(0))
            self.heads(self.encoder.feature_dict["readout"])

    def _predict_batch(self, structures: list[Structure]) -> dict:
        """Run the GNN on a batch of structures, return de-normalized predictions."""
        graphs, lats, state_attrs = [], [], []
        for s in structures:
            g, lat, state_attr = self.converter.get_graph(s)
            lat0 = lat[0] if lat.dim() == 3 else lat
            g.pos = g.frac_coords @ lat0
            g.pbc_offshift = torch.matmul(g.pbc_offset, lat0)
            graphs.append(g)
            lats.append(lat)
            state_attrs.append(torch.as_tensor(state_attr, dtype=torch.float32))

        from torch_geometric.data import Batch
        batch_g = Batch.from_data_list(graphs).to(DEVICE)
        batch_state = torch.stack(state_attrs).to(DEVICE)

        with torch.no_grad():
            _ = self.encoder(batch_g, batch_state)
            reg_pred, cls_pred = self.heads(self.encoder.feature_dict["readout"])

        out = {}
        for j, t in enumerate(REG_KEYS):
            mean, std = self.stats[t]
            out[t] = (reg_pred[:, j].cpu().numpy() * std + mean)

        for t in CLS_KEYS:
            probs = torch.softmax(cls_pred[t], dim=1).cpu().numpy()
            idx = probs.argmax(axis=1)
            if t == "ordering":
                out[t] = [ORDERING_CLASSES[i] for i in idx]
            else:
                out[t] = idx.astype(bool)
            out[f"{t}__confidence"] = probs.max(axis=1)

        return out

    def _predict_density_baseline(self, structures: list[Structure]) -> np.ndarray:
        comps = [s.composition for s in structures]
        feats = np.array(self.magpie.featurize_many(comps, ignore_errors=True), dtype=float)
        return self.baseline_density_model.predict(feats)

    def score(self, structures: list[Structure], thresholds: dict = None,
              batch_size: int = 32) -> pd.DataFrame:
        thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        rows = []

        for i in range(0, len(structures), batch_size):
            chunk = structures[i:i + batch_size]
            preds = self._predict_batch(chunk)
            for j, s in enumerate(chunk):
                row = {"formula": s.composition.reduced_formula}
                for k, v in preds.items():
                    row[k] = v[j]
                rows.append(row)

        df = pd.DataFrame(rows)

        if self.baseline_density_model is not None:
            df["density"] = self._predict_density_baseline(structures)
            df["density_source"] = "baseline_xgb"
        else:
            df["density_source"] = "gnn"
            
        passes = pd.Series(True, index=df.index)
        reasons = pd.Series("", index=df.index)

        hull_fail = df["energy_above_hull"] > thresholds["max_energy_above_hull"]
        passes &= ~hull_fail
        reasons = reasons.where(~hull_fail, reasons + "energy_above_hull_too_high;")

        if thresholds["require_is_stable"]:
            stable_fail = ~df["is_stable"]
            passes &= ~stable_fail
            reasons = reasons.where(~stable_fail, reasons + "not_predicted_stable;")

        if thresholds["band_gap_range"] is not None:
            lo, hi = thresholds["band_gap_range"]
            gap_fail = (df["band_gap"] < lo) | (df["band_gap"] > hi)
            passes &= ~gap_fail
            reasons = reasons.where(~gap_fail, reasons + "band_gap_out_of_range;")

        df["screen_pass"] = passes
        df["screen_fail_reasons"] = reasons
        return df


def load_structures_from_file(path: Path) -> list[Structure]:
    """Accepts a .json/.jsonl of pymatgen Structure.as_dict() records."""
    structures = []
    text = path.read_text()
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                structures.append(Structure.from_dict(json.loads(line)))
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else [data]
        for rec in records:
            structures.append(Structure.from_dict(rec))
    return structures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", required=True, type=Path,
                     help=".json or .jsonl file of pymatgen Structure dicts to screen")
    ap.add_argument("--gnn-checkpoint", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                     help="Directory with density_xgb.joblib from baseline_composition_model.py --model-dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-energy-above-hull", type=float, default=0.05)
    ap.add_argument("--require-stable", action="store_true")
    ap.add_argument("--band-gap-min", type=float, default=None)
    ap.add_argument("--band-gap-max", type=float, default=None)
    args = ap.parse_args()

    thresholds = {
        "max_energy_above_hull": args.max_energy_above_hull,
        "require_is_stable": args.require_stable,
        "band_gap_range": (
            (args.band_gap_min, args.band_gap_max)
            if args.band_gap_min is not None and args.band_gap_max is not None
            else None
        ),
    }

    print(f"Loading structures from {args.structures}...")
    structures = load_structures_from_file(args.structures)
    print(f"  {len(structures)} structures to screen")

    screener = StackScreener(
        gnn_checkpoint=str(args.gnn_checkpoint),
        baseline_dir=str(args.baseline_dir) if args.baseline_dir else None,
    )
    df = screener.score(structures, thresholds=thresholds)

    n_pass = df["screen_pass"].sum()
    print(f"\n{n_pass}/{len(df)} structures passed screening")
    df.to_csv(args.out, index=False)
    print(f"Wrote full results to {args.out}")


if __name__ == "__main__":
    main()
