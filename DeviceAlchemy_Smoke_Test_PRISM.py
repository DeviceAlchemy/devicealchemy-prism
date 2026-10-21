"""
PRISM v1.0 — Smoke Test
Developed by: DeviceAlchemy
Copyright (c) 2026 DeviceAlchemy LLC
Code author: Shehrin Sayed, Ph.D.
SPDX-License-Identifier: AGPL-3.0-only
"""

from pymatgen.core import Structure, Lattice
from DeviceAlchemy_PRISM_inference import StackScreener

screener = StackScreener(gnn_checkpoint="MODEL/PRISM.pt")

nacl = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
mgo  = Structure(Lattice.cubic(4.21), ["Mg", "O"],  [[0, 0, 0], [0.5, 0.5, 0.5]])
gaas = Structure(Lattice.cubic(5.65), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]])
si   = Structure(Lattice.cubic(5.43), ["Si"] * 8,
                 [[0, 0, 0], [0.25, 0.25, 0.25], [0, 0.5, 0.5], [0.25, 0.75, 0.75],
                  [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0.5, 0.5, 0], [0.75, 0.75, 0.25]])

result = screener.score([nacl, mgo, si, gaas])
print(result[["formula", "energy_above_hull", "band_gap", "is_metal",
              "is_stable", "ordering", "screen_pass"]].to_string())