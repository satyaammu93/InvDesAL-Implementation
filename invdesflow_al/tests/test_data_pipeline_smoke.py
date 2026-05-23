"""Smoke test for the pretraining data pipeline (REBUILD_PLAN.md sec. 6).

Dependency-free: builds a synthetic JSONL corpus, then exercises
load -> filter -> dedup -> diversity-sample -> exclude -> dataloader ->
one Algorithm-1 step on a tiny generator.

Run:  python -m invdesflow_al.tests.test_data_pipeline_smoke
"""

from __future__ import annotations

import copy
import os
import random
import tempfile

import torch

from ..data.datasets import (
    StructureRecord,
    dedup_records,
    filter_records,
    load_structures,
    write_manifest,
)
from ..data.diversity import DiversitySampler
from ..data.torch_dataset import (
    apply_exclusion,
    exclusion_keyset,
    make_dataloaders,
)
from ..models.generator import CrystalGenerator, load_config


def _synth(n: int, seed: int = 0) -> list[StructureRecord]:
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        na = rng.randint(2, 8)
        z = [rng.randint(1, 12) for _ in range(na)]
        frac = [[rng.random() for _ in range(3)] for _ in range(na)]
        lat = [[4.0 + rng.uniform(-0.3, 0.3) if i == j else rng.uniform(-0.2, 0.2)
                for j in range(3)] for i in range(3)]
        recs.append(StructureRecord(
            z=z, frac=frac, lattice=lat,
            e_form=rng.uniform(-4.0, 0.5),
            e_hull=rng.uniform(0.0, 0.2),
            spacegroup=rng.choice([1, 2, 62, 139, 225]),
            source="",  # left empty -> loader tags provenance
        ))
    return recs


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        # write two synthetic corpora + a tiny "CSP test set" to exclude
        a_path = os.path.join(d, "alex.jsonl")
        g_path = os.path.join(d, "gnome.jsonl")
        x_path = os.path.join(d, "csp_testset.jsonl")
        write_manifest(_synth(400, 1), a_path)
        write_manifest(_synth(300, 2), g_path)
        excl_recs = _synth(20, 3)
        write_manifest(excl_recs, x_path)

        raw = list(load_structures(a_path, "alex-mp-20")) + \
              list(load_structures(g_path, "gnome"))
        assert len(raw) == 700
        assert {r.source for r in raw} == {"alex-mp-20", "gnome"}

        # add guaranteed oversize + duplicate to prove filter/dedup bite
        big = copy.deepcopy(raw[0]); big.z = list(range(1, 30)); big.frac = [[0.0]*3]*29
        big.lattice = raw[0].lattice
        raw.append(big)
        raw.append(copy.deepcopy(raw[0]))  # exact dup

        filt = list(filter_records(raw, max_atoms=20, e_form_max=0.0))
        assert all(r.num_atoms <= 20 and r.e_form is not None and r.e_form <= 0.0
                   for r in filt)
        print(f"filter: {len(raw)} -> {len(filt)} (<=20 atoms, e_form<=0)")

        dedup = list(dedup_records(filt))
        print(f"dedup : {len(filt)} -> {len(dedup)}")

        excl = exclusion_keyset(list(load_structures(x_path)))
        kept = apply_exclusion(dedup, excl)
        print(f"exclude CSP test set: {len(dedup)} -> {len(kept)}")

        sampler = DiversitySampler(mode="round_robin", seed=0)
        rep_before = sampler.coverage_report(kept)
        sel = sampler.select(kept, target_size=min(150, len(kept)))
        rep_after = sampler.coverage_report(sel)
        print(f"diversity: {rep_before} -> {rep_after}")
        # round-robin must not collapse coverage
        assert rep_after["n_buckets"] >= min(rep_before["n_buckets"], len(sel))

        cfg = copy.deepcopy(load_config())
        cfg["model"].update(hidden_dim=32, num_gnn_layers=2, num_atom_types=12,
                             fourier_freqs=4, time_embed_dim=16)
        cfg["diffusion"]["num_steps"] = 100
        cfg["data"].update(batch_size_train=16, batch_size_val=8,
                            batch_size_test=8, num_preprocess_workers=0)

        train_dl, val_dl, test_dl = make_dataloaders(sel, cfg)
        assert len(train_dl) > 0 and len(val_dl) >= 0
        gen = CrystalGenerator(cfg, device="cpu")
        batch = next(iter(train_dl))
        loss = gen.loss_on_batch(batch)["total"]
        assert torch.isfinite(loss), "non-finite loss from dataloader batch"
        print(f"dataloader -> Algorithm 1 loss = {float(loss):.4f}  "
              f"(train/val/test batches = {len(train_dl)}/{len(val_dl)}/{len(test_dl)})")

    print("DATA PIPELINE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
