"""
PRISM v1.0 — Training
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
import torch.nn as nn
from torch_geometric.data import Batch

import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader

from DeviceAlchemy_PRISM_Data_Preparation import (
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGETS,
    structures_from_df,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_CARDINALITY = {"is_metal": 2, "is_stable": 2, "ordering": 4}
ORDERING_CLASSES = ["NM", "FM", "AFM", "FiM"] 

REG_KEYS = list(REGRESSION_TARGETS)
REG_MASK_KEYS = [f"{t}__mask" for t in REGRESSION_TARGETS]
CLS_KEYS = list(CLASSIFICATION_TARGETS)
CLS_MASK_KEYS = [f"{t}__mask" for t in CLASSIFICATION_TARGETS]
ALL_LABEL_KEYS = REG_KEYS + REG_MASK_KEYS + CLS_KEYS + CLS_MASK_KEYS


def compute_target_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for t in REGRESSION_TARGETS:
        vals = df.loc[df[f"{t}__mask"] == 1, t].astype(float)
        mean = vals.mean() if len(vals) else 0.0
        std = vals.std() if len(vals) > 1 else 1.0
        std = std if std > 1e-8 else 1.0
        stats[t] = (mean, std)
    return stats


def build_labels_dict(df: pd.DataFrame, stats: dict) -> dict:
    """Pack normalized targets, class indices and masks as MGLDataset labels (ALL_LABEL_KEYS order)."""
    labels = {}
    for t in REGRESSION_TARGETS:
        mean, std = stats[t]
        mask = df[f"{t}__mask"].values
        raw = df[t].fillna(0.0).astype(float).values
        norm = np.where(mask == 1, (raw - mean) / std, 0.0)
        labels[t] = norm.tolist()
        labels[f"{t}__mask"] = mask.astype(float).tolist()

    for t in CLASSIFICATION_TARGETS:
        mask = df[f"{t}__mask"].values.astype(float)
        if t == "ordering":
            idx = df[t].apply(
                lambda v: ORDERING_CLASSES.index(v) if v in ORDERING_CLASSES else 0
            ).values.astype(float)
        else:
            idx = df[t].fillna(False).astype(bool).astype(float).values
        labels[t] = idx.tolist()
        labels[f"{t}__mask"] = mask.tolist()
    return labels


def multitask_collate(batch: list):
    """
    Custom collate: stacks all label keys (matgl keeps only one) and
    computes per-graph g.pos / g.pbc_offshift before batching.
    """
    graphs, lattices, state_attrs, label_dicts = zip(*batch)

    for g, lat in zip(graphs, lattices):
        lat0 = lat[0] if lat.dim() == 3 else lat
        g.pos = g.frac_coords @ lat0
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat0)

    g = Batch.from_data_list(list(graphs))
    lat = lattices[0] if len(lattices) == 1 else torch.squeeze(torch.stack(list(lattices)))
    state_attr = torch.stack(list(state_attrs))

    label_matrix = torch.tensor(
        [[d[k] for k in ALL_LABEL_KEYS] for d in label_dicts],
        dtype=torch.float32,
    )
    return g, lat, state_attr, label_matrix


def unpack_labels(label_matrix: torch.Tensor):
    """Slice the collated (batch, n_keys) tensor back into named pieces."""
    n_reg = len(REG_KEYS)
    n_cls = len(CLS_KEYS)
    i = 0
    reg_vals = label_matrix[:, i:i + n_reg]; i += n_reg
    reg_mask = label_matrix[:, i:i + n_reg]; i += n_reg
    cls_vals = label_matrix[:, i:i + n_cls]; i += n_cls
    cls_mask = label_matrix[:, i:i + n_cls]; i += n_cls
    return reg_vals, reg_mask, cls_vals, cls_mask


class MultiTaskHeads(nn.Module):
    """
    Multi-task heads on M3GNet's readout embedding.
    LazyLinear infers the input width at first forward pass.
    """
    def __init__(self, n_reg: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LazyLinear(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.reg_head = nn.Linear(128, n_reg)
        self.cls_heads = nn.ModuleDict({
            t: nn.Linear(128, CLASS_CARDINALITY[t]) for t in CLASSIFICATION_TARGETS
        })

    def forward(self, readout_vec):
        h = self.trunk(readout_vec)
        reg_out = self.reg_head(h)
        cls_out = {t: head(h) for t, head in self.cls_heads.items()}
        return reg_out, cls_out


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


def run_epoch(encoder, heads, loader, optimizer=None):
    train = optimizer is not None
    encoder.train(train)
    heads.train(train)
    total_loss, n = 0.0, 0

    for g, lat, state_attr, label_matrix in loader:
        g = g.to(DEVICE)
        state_attr = state_attr.to(DEVICE)
        label_matrix = label_matrix.to(DEVICE)
        reg_vals, reg_mask, cls_vals, cls_mask = unpack_labels(label_matrix)

        with torch.set_grad_enabled(train):
            _ = encoder(g, state_attr)
            readout = encoder.feature_dict["readout"]
            reg_pred, cls_pred = heads(readout)

            loss = masked_mse(reg_pred, reg_vals, reg_mask)
            for j, t in enumerate(CLS_KEYS):
                labels = cls_vals[:, j].long()
                mask = cls_mask[:, j]
                if mask.sum() == 0:
                    continue
                ce = nn.functional.cross_entropy(cls_pred[t], labels, reduction="none")
                loss = loss + (ce * mask).sum() / mask.sum().clamp(min=1.0)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = state_attr.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepdir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pretrained", default="M3GNet-Eform-MP-2019.4.1")
    ap.add_argument("--resume", type=Path, default=None,
                     help="Path to a checkpoint .pt to resume from (e.g. model_out/final_model.pt). "
                          "Skips loading --pretrained's fresh weights.")
    ap.add_argument("--start-epoch", type=int, default=0,
                     help="Epoch number to start counting from when resuming (for correct checkpoint naming/logs).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained encoder: {args.pretrained}")
    encoder = matgl.load_model(args.pretrained).to(DEVICE)

    converter = Structure2Graph(
        element_types=encoder.element_types, cutoff=encoder.cutoff
    )

    train_df = pd.read_parquet(args.prepdir / "train.parquet")
    val_df = pd.read_parquet(args.prepdir / "val.parquet")

    stats = compute_target_stats(train_df)
    print("Target normalization (mean, std):")
    for t, (m, s) in stats.items():
        print(f"  {t:45s} mean={m:.4f} std={s:.4f}")

    print("Building structures + graph datasets (this converts every "
          "structure to a graph — slow on first run, cached after)...")
    train_structs = structures_from_df(train_df)
    val_structs = structures_from_df(val_df)

    train_ds = MGLDataset(
        structures=train_structs,
        labels=build_labels_dict(train_df, stats),
        converter=converter,
        root=str(args.out / "mgl_train_cache"),
    )
    val_ds = MGLDataset(
        structures=val_structs,
        labels=build_labels_dict(val_df, stats),
        converter=converter,
        root=str(args.out / "mgl_val_cache"),
    )

    train_loader, val_loader = MGLDataLoader(
        train_data=train_ds,
        val_data=val_ds,
        collate_fn=multitask_collate,
        batch_size=args.batch_size,
    )

    heads = MultiTaskHeads(n_reg=len(REG_KEYS)).to(DEVICE)
    
    with torch.no_grad():
        g0, lat0, state_attr0, _ = next(iter(train_loader))
        g0 = g0.to(DEVICE)
        _ = encoder(g0, state_attr0.to(DEVICE))
        heads(encoder.feature_dict["readout"])

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        heads.load_state_dict(ckpt["heads"])

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(heads.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )

    print(f"\nTraining on {DEVICE} for {args.epochs} epochs (starting at epoch {args.start_epoch})...")
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        train_loss = run_epoch(encoder, heads, train_loader, optimizer)
        val_loss = run_epoch(encoder, heads, val_loader, optimizer=None)
        print(f"epoch {epoch:4d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if epoch % 5 == 0 or epoch == args.start_epoch + args.epochs - 1:
            torch.save(
                {"encoder": encoder.state_dict(), "heads": heads.state_dict()},
                args.out / f"checkpoint_epoch{epoch}.pt",
            )

    torch.save(
        {"encoder": encoder.state_dict(), "heads": heads.state_dict()},
        args.out / "final_model.pt",
    )
    np.save(args.out / "target_stats.npy", stats, allow_pickle=True)
    print(f"Saved final model + target stats to {args.out}")


if __name__ == "__main__":
    main()
