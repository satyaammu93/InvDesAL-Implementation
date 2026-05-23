"""Quick eval for iteration cycles: ~10-15 min, samples 512 by default,
collects A-channel diagnostics during sampling, and writes a compact JSON.

Reports:
  unique rate (formula diversity)
  lattice sanity (vpa percentiles, sane fraction, nan/inf)
  atom-output stats (entropy, top-element fraction)
  A-channel sampler diagnostics (per-step max/median |A|, saturated fraction,
                                  first saturation timestep)

Usage:
  python -m invdesflow_al.scripts.debug_eval_quick \
      --ckpt gen_10k.ckpt --manifest data_raw/pretrain_10k.jsonl \
      --n-sample 512 --gen-batch 256 --device cuda --out eval_quick.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from functools import reduce
from math import gcd

import torch

from ..data.datasets import load_structures
from ..models.generator import CrystalGenerator


def reduced_formula(z):
    c = Counter(z)
    g = reduce(gcd, c.values()) if c else 1
    return "-".join(f"{e}{n // g}" for e, n in sorted(c.items()))


def atom_count_hist(manifest, cap=20000):
    hist = []
    for i, r in enumerate(load_structures(manifest)):
        hist.append(len(r.z))
        if i + 1 >= cap:
            break
    return hist


def entropy(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in counter.values() if c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-sample", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_quick.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gen = CrystalGenerator(ck["cfg"], device=args.device)
    gen.load_state_dict(ck["state_dict"])
    gen.eval()
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch','?')}, "
          f"val {ck.get('val_loss','?')})", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    hist_t = torch.tensor(atom_count_hist(args.manifest))

    formulas: list[str] = []
    vpa: list[float] = []
    elems: Counter = Counter()
    nan = 0
    # accumulate per-step A stats across batches
    max_a_per_t: dict[int, list[float]] = {}
    med_a_per_t: dict[int, list[float]] = {}
    sat_per_t: dict[int, list[float]] = {}

    t0 = time.time()
    while len(formulas) < args.n_sample:
        bs = min(args.gen_batch, args.n_sample - len(formulas))
        idx = torch.randint(0, len(hist_t), (bs,), generator=g)
        n_atoms = hist_t[idx].tolist()
        stats: dict = {}
        crystals = gen.sample(num_atoms=n_atoms, stats=stats)
        for c in crystals:
            zs = c.atom_types.tolist()
            formulas.append(reduced_formula(zs))
            elems.update(zs)
            if torch.isfinite(c.lattice).all():
                vpa.append(float(torch.det(c.lattice).abs()) / c.num_atoms)
            else:
                nan += 1
        for k, m_a, md_a, sf in zip(
            stats["t"], stats["max_a"], stats["med_a"], stats["sat_frac"]
        ):
            max_a_per_t.setdefault(k, []).append(m_a)
            med_a_per_t.setdefault(k, []).append(md_a)
            sat_per_t.setdefault(k, []).append(sf)
        print(f"  generated {len(formulas)}/{args.n_sample} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # unique rate at n=len(formulas)
    n = len(formulas)
    uniq = len(set(formulas)) / n
    # lattice sanity
    vt = torch.tensor(vpa) if vpa else torch.zeros(1)
    qf = lambda f: float(vt.sort().values[min(len(vt) - 1, int(f * len(vt)))])
    sane = float(((vt > 0) & (vt <= 500)).float().mean())
    # atom output
    top_z, top_n = elems.most_common(1)[0] if elems else (0, 0)
    top_frac = top_n / max(sum(elems.values()), 1)
    # A diagnostics: aggregate per t (mean across batches), and overall worst
    ts = sorted(max_a_per_t)
    agg_max = {t: max(max_a_per_t[t]) for t in ts}
    agg_med = {t: sum(med_a_per_t[t]) / len(med_a_per_t[t]) for t in ts}
    agg_sat = {t: max(sat_per_t[t]) for t in ts}
    worst_max_a = max(agg_max.values())
    worst_sat = max(agg_sat.values())
    first_sat_t = next((t for t in sorted(ts, reverse=True) if agg_sat[t] > 0), None)

    out = {
        "ckpt": args.ckpt,
        "epoch": ck.get("epoch"),
        "val_loss": ck.get("val_loss"),
        "n_sample": n,
        "unique_rate": round(uniq, 4),
        "distinct_formulas": len(set(formulas)),
        "lattice": {
            "vpa_min": round(float(vt.min()), 2),
            "vpa_p5": round(qf(0.05), 2),
            "vpa_median": round(float(vt.median()), 2),
            "vpa_p95": round(qf(0.95), 2),
            "vpa_max": round(float(vt.max()), 2),
            "sane_fraction": round(sane, 3),
            "nan": nan,
        },
        "atom_output": {
            "n_elements": len(elems),
            "entropy_nats": round(entropy(elems), 3),
            "top_z": top_z,
            "top_fraction": round(top_frac, 3),
        },
        "a_channel": {
            "worst_max_abs_A": round(worst_max_a, 2),
            "worst_saturated_fraction": round(worst_sat, 4),
            "first_saturation_t": first_sat_t,   # None if never saturated
            "max_abs_A_at_t999": round(agg_max.get(999, 0), 2),
            "max_abs_A_at_t500": round(agg_max.get(500, 0), 2),
            "max_abs_A_at_t100": round(agg_max.get(100, 0), 2),
            "max_abs_A_at_t0":   round(agg_max.get(0, 0), 2),
            "sat_frac_at_t999": round(agg_sat.get(999, 0), 4),
            "sat_frac_at_t500": round(agg_sat.get(500, 0), 4),
            "sat_frac_at_t100": round(agg_sat.get(100, 0), 4),
        },
    }

    print()
    print(f"unique rate         : {out['unique_rate']}  ({out['distinct_formulas']}/{n})")
    L = out["lattice"]
    print(f"lattice vpa min/p5/med/p95/max : {L['vpa_min']} / {L['vpa_p5']} "
          f"/ {L['vpa_median']} / {L['vpa_p95']} / {L['vpa_max']}")
    print(f"sane fraction       : {L['sane_fraction']}   nan: {L['nan']}")
    A = out["atom_output"]
    print(f"atom output         : {A['n_elements']} elements   entropy "
          f"{A['entropy_nats']}   top Z={A['top_z']} ({A['top_fraction']:.1%})")
    AC = out["a_channel"]
    print(f"A-channel worst     : max|A|={AC['worst_max_abs_A']}  "
          f"sat_frac={AC['worst_saturated_fraction']}  "
          f"first_sat_t={AC['first_saturation_t']}")
    print(f"A at t=999/500/100/0: max|A|={AC['max_abs_A_at_t999']}/"
          f"{AC['max_abs_A_at_t500']}/{AC['max_abs_A_at_t100']}/"
          f"{AC['max_abs_A_at_t0']}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    print("RESULT_JSON " + json.dumps({
        "unique_rate": out["unique_rate"],
        **L,
        "first_sat_t": AC["first_saturation_t"],
        "worst_sat": AC["worst_saturated_fraction"],
        "worst_max_A": AC["worst_max_abs_A"],
        "top_z": A["top_z"], "top_frac": A["top_fraction"],
    }))


if __name__ == "__main__":
    main()
