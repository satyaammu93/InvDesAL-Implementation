"""Lazy torch Dataset/DataLoader over StructureRecords.

Batch sizes 96/64/64 (train/val/test) from Table S.2. Structures are kept as
lightweight records and materialized into `Crystal` only on __getitem__, so a
~1M-structure corpus stays cheap in RAM.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .batch import collate
from .datasets import StructureRecord, _reduced_formula_key
from .representation import Crystal


class CrystalDataset(Dataset):
    def __init__(self, records: Sequence[StructureRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Crystal:
        return self.records[i].to_crystal()


def make_collate(cutoff: float, max_nbr: int, mode: str = "complete"):
    def _c(crystals: list[Crystal]):
        return collate(crystals, cutoff, max_nbr, mode)

    return _c


def split_records(
    records: Sequence[StructureRecord],
    ratios=(0.95, 0.025, 0.025),
    seed: int = 0,
) -> tuple[list, list, list]:
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    pick = lambda s: [records[i] for i in s]
    return pick(idx[:n_tr]), pick(idx[n_tr : n_tr + n_va]), pick(idx[n_tr + n_va :])


def exclusion_keyset(records: Sequence[StructureRecord]) -> set:
    """Keys for held-out CSP test sets (Perov-5 / MP-20 / MPTS-52).

    Paper: "rigorous exclusion of test sets ... during data preprocessing."
    Cheap key = (reduced formula, spacegroup); pass these to
    `apply_exclusion` over the pretraining pool.
    """
    return {(_reduced_formula_key(r.z), r.spacegroup) for r in records}


def apply_exclusion(
    records: Sequence[StructureRecord], excluded: set
) -> list[StructureRecord]:
    return [
        r for r in records if (_reduced_formula_key(r.z), r.spacegroup) not in excluded
    ]


def make_dataloaders(records: Sequence[StructureRecord], cfg: dict):
    d = cfg["data"]
    cutoff = float(d["neighbor_cutoff_radius"])
    max_nbr = int(d["max_neighbors_per_atom"])
    workers = int(d.get("num_preprocess_workers", 0))
    coll = make_collate(cutoff, max_nbr, str(d.get("graph_construction", "complete")))

    tr, va, te = split_records(records)
    mk = lambda recs, bs, sh: DataLoader(
        CrystalDataset(recs),
        batch_size=bs,
        shuffle=sh,
        num_workers=workers,
        collate_fn=coll,
        drop_last=sh,
    )
    return (
        mk(tr, int(d["batch_size_train"]), True),
        mk(va, int(d["batch_size_val"]), False),
        mk(te, int(d["batch_size_test"]), False),
    )
