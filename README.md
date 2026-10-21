# PRISM — Property Reasoning & Interoperability for Stack Matching

PRISM is a multi-task graph neural network from [DeviceAlchemy.ai](https://devicealchemy.ai) that predicts bulk materials properties from crystal structure. It's built on a fine-tuned M3GNet encoder and partially trained on Materials Project data under the CC BY 4.0 license.**.

## What it predicts

| Type | Properties |
|---|---|
| Regression | Formation energy per atom, energy above hull, band gap, density, normalized magnetization |
| Classification | Metal / non-metal, stable / unstable, magnetic ordering (NM / FM / AFM / FiM) |

## Quick start

```bash
pip install -r requirements.txt
python DeviceAlchemy_Smoke_Test_PRISM.py
```

## Usage

```python
from DeviceAlchemy_PRISM_inference import StackScreener

screener = StackScreener(gnn_checkpoint="MODEL/PRISM.pt")
results = screener.score(list_of_pymatgen_structures)
print(results)
```

Screen a batch of structures from the command line:

```bash
python DeviceAlchemy_PRISM_inference.py \
    --structures candidates.jsonl \
    --gnn-checkpoint MODEL/PRISM.pt \
    --out screened.csv
```

## Repository contents

| File | Purpose |
|---|---|
| `DeviceAlchemy_PRISM_Data_Preparation.py` | Builds targets, missing-value masks, and chemsys-based train/val/test splits |
| `DeviceAlchemy_PRISM_training.py` | Trains the multi-task GNN |
| `DeviceAlchemy_PRISM_Evaluate_GNN.py` | Reports MAE and accuracy on held-out data |
| `DeviceAlchemy_PRISM_inference.py` | `StackScreener` prediction API and CLI |
| `DeviceAlchemy_Smoke_Test_PRISM.py` | Quick end-to-end check |
| `check_gnome_sources.py` | Verifies the training data contains no GNoME (CC BY-NC) records |
| `MODEL/` | Trained weights (`PRISM.pt`) and normalization statistics |

## Reproducing the model

```bash
python DeviceAlchemy_PRISM_Data_Preparation.py --input mp_summary.jsonl --outdir prepped/
python DeviceAlchemy_PRISM_training.py --prepdir prepped/ --out MODEL/
python DeviceAlchemy_PRISM_Evaluate_GNN.py --prepdir prepped/ --model-dir MODEL/ --checkpoint PRISM.pt --split test
```

Splits are made by chemical system, not by row, so no chemistry appears in both training and test sets.

## Limitations

PRISM predicts **bulk, single-material** properties. It does not model interfaces, transport, or spin-orbit effects, and its predictions are a screening step, not a substitute for DFT.

## Data attribution

Trained on data from the [Materials Project](https://materialsproject.org), used under CC BY 4.0.

> A. Jain et al., "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation," *APL Materials* 1, 011002 (2013).

## License

AGPL-3.0. For commercial licensing, contact DeviceAlchemy at **support@devicealchemy.ai**.

© 2026 DeviceAlchemy LLC — Shehrin Sayed, Ph.D.
