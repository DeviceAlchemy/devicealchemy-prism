"""
PRISM v1.0 — Evaluation
Developed by: DeviceAlchemy
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
SPDX-License-Identifier: AGPL-3.0-only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader

from DeviceAlchemy_PRISM_Data_Preparation import REGRESSION_TARGETS, CLASSIFICATION_TARGETS, structures_from_df
from DeviceAlchemy_PRISM_training import (
    MultiTaskHeads,
    build_labels_dict,
    multitask_collate,
    unpack_labels,
    REG_KEYS,
    CLS_KEYS,
    ORDERING_CLASSES,
    DEVICE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepdir", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pretrained", default="M3GNet-Eform-MP-2019.4.1")
    ap.add_argument("--checkpoint", default="final_model.pt",
                     help="Checkpoint filename inside --model-dir to load (default: final_model.pt). "
                          "Use e.g. checkpoint_epoch15.pt to evaluate a specific earlier epoch.")
    args = ap.parse_args()

    stats = np.load(args.model_dir / "target_stats.npy", allow_pickle=True).item()

    print(f"Loading pretrained encoder architecture: {args.pretrained}")
    encoder = matgl.load_model(args.pretrained).to(DEVICE)
    converter = Structure2Graph(element_types=encoder.element_types, cutoff=encoder.cutoff)

    heads = MultiTaskHeads(n_reg=len(REG_KEYS)).to(DEVICE)

    ckpt = torch.load(args.model_dir / args.checkpoint, map_location=DEVICE)

    df = pd.read_parquet(args.prepdir / f"{args.split}.parquet")
    structs = structures_from_df(df)

    ds = MGLDataset(
        structures=structs,
        labels=build_labels_dict(df, stats),
        converter=converter,
        root=str(args.model_dir / f"mgl_eval_{args.split}_cache"),
    )
    loader, _ = MGLDataLoader(
        train_data=ds, val_data=ds,
        collate_fn=multitask_collate,
        batch_size=args.batch_size,
    )
    
    with torch.no_grad():
        g0, lat0, state_attr0, _ = next(iter(loader))
        _ = encoder(g0.to(DEVICE), state_attr0.to(DEVICE))
        heads(encoder.feature_dict["readout"])

    encoder.load_state_dict(ckpt["encoder"])
    heads.load_state_dict(ckpt["heads"])
    encoder.eval()
    heads.eval()

    all_reg_pred, all_reg_true, all_reg_mask = [], [], []
    all_cls_pred = {t: [] for t in CLS_KEYS}
    all_cls_true = {t: [] for t in CLS_KEYS}
    all_cls_mask = {t: [] for t in CLS_KEYS}

    with torch.no_grad():
        for g, lat, state_attr, label_matrix in loader:
            g = g.to(DEVICE)
            state_attr = state_attr.to(DEVICE)
            reg_vals, reg_mask, cls_vals, cls_mask = unpack_labels(label_matrix)

            _ = encoder(g, state_attr)
            reg_pred, cls_pred = heads(encoder.feature_dict["readout"])

            all_reg_pred.append(reg_pred.cpu().numpy())
            all_reg_true.append(reg_vals.numpy())
            all_reg_mask.append(reg_mask.numpy())

            for j, t in enumerate(CLS_KEYS):
                pred_idx = cls_pred[t].argmax(dim=1).cpu().numpy()
                all_cls_pred[t].append(pred_idx)
                all_cls_true[t].append(cls_vals[:, j].numpy())
                all_cls_mask[t].append(cls_mask[:, j].numpy())

    reg_pred = np.concatenate(all_reg_pred, axis=0)
    reg_true = np.concatenate(all_reg_true, axis=0)
    reg_mask = np.concatenate(all_reg_mask, axis=0)

    print(f"\n{'target':45s} {'MAE (' + args.split + ')':>14s}   n_eval")
    print("-" * 70)
    for j, t in enumerate(REG_KEYS):
        mean, std = stats[t]
        m = reg_mask[:, j] == 1
        if m.sum() < 5:
            print(f"{t:45s} {'skip (too few labeled rows)':>14s}")
            continue
        
        pred_real = reg_pred[m, j] * std + mean
        true_real = reg_true[m, j] * std + mean
        mae = np.abs(pred_real - true_real).mean()
        print(f"{t:45s} {mae:14.4f}   {int(m.sum())}")

    print()
    for t in CLS_KEYS:
        pred = np.concatenate(all_cls_pred[t])
        true = np.concatenate(all_cls_true[t]).astype(int)
        mask = np.concatenate(all_cls_mask[t]) == 1
        if mask.sum() < 5:
            print(f"{t:45s} skip (too few labeled rows)")
            continue
        acc = (pred[mask] == true[mask]).mean()
        print(f"{t:45s} accuracy={acc:.3f}   n_eval={int(mask.sum())}")


if __name__ == "__main__":
    main()
