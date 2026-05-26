"""Score-movement test for Stage-0 AL (Entry 14 v2 / Entry 16).

Closes the AL loop: fine-tune a copy of the generator on the CHGNet-RELAXED
selected structures from a Stage-0 round, then test whether a fresh held-out
generated batch — never seen by the fine-tune — shifts in the intended
direction.

Pass criteria (with the Entry 14 v2 direction-correction codified in Entry 16):
  1. Safety. Post-finetune `debug_eval_quick` meets the Entry-8 thresholds
     (unique_rate >= 0.5, sane_fraction >= 0.95, vpa_median in [5, 100],
     first_sat_t is None, nan == 0).
  2. delta_e DROPS. median(delta_e) on a fresh held-out CHGNet-scored batch
     is lower than the round-0 baseline by at least --delta-thresh (default
     0.05 eV/atom). Less relaxation depth = generator outputs are closer to
     equilibrium = AL has moved the distribution toward stability.
  3. Convergence holds. fraction(converged_ml) post does not drop more than
     --conv-tol (default 0.10) vs pre.
  4. Not memorization. Fraction of post-valid formulas whose reduced formula
     was in the fine-tune-seen set is <= --max-memo (default 0.50).

Usage:
  python -m invdesflow_al.scripts.run_al_score_movement \
      --ckpt checkpoints/gen_150k.ckpt \
      --round0-dir al_runs/chgnet_stage0_round0 \
      --manifest data_raw/pretrain.jsonl \
      --finetune-steps 500 \
      --post-num-generate 500 --post-max-relax 200 \
      --device cuda \
      --out-dir al_runs/chgnet_stage0_round0_movement
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from ..al import CHGNetOracle, manifest_novelty_set, novelty_key
from ..data.representation import Crystal
from ..models.generator import CrystalGenerator
from .run_tiny_al_dryrun import (
    atom_count_hist,
    element_set_key,
    reduced_formula,
    validity,
)


def _load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _quant(arr, ndigits: int = 4):
    import numpy as np

    return tuple(round(float(x), ndigits) for x in np.asarray(arr).flatten())


def _match_key(z, frac, lattice):
    return (tuple(int(x) for x in z), _quant(frac), _quant(lattice))


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def _element_histogram(crystal_atom_lists, top_n: int = 20) -> dict:
    """Element histogram + top-N over a list of atomic-number lists."""
    from collections import Counter

    c: Counter = Counter()
    for zs in crystal_atom_lists:
        c.update(int(z) for z in zs)
    total = sum(c.values()) or 1
    top = sorted(c.items(), key=lambda kv: -kv[1])[:top_n]
    return {
        "total_atoms": total,
        "distinct_elements": len(c),
        "top": [{"z": int(z), "count": int(n),
                 "fraction": round(n / total, 4)} for z, n in top],
        "fractions_all": {int(z): round(n / total, 5) for z, n in c.items()},
    }


def _baseline_element_fractions(manifest_path: str, cache_path: Path) -> dict:
    """Manifest-wide per-element fractions, cached after first computation."""
    if cache_path.exists():
        d = json.loads(cache_path.read_text())
        return {int(k): float(v) for k, v in d.items()}
    from collections import Counter

    from ..data.datasets import load_structures

    c: Counter = Counter()
    for r in load_structures(manifest_path):
        c.update(int(z) for z in r.z)
    total = sum(c.values()) or 1
    out = {int(z): n / total for z, n in c.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def _enrichment(hist: dict, baseline: dict) -> list:
    """Per top-Z enrichment: post-fraction / manifest-baseline-fraction."""
    out = []
    for entry in hist["top"]:
        z = entry["z"]
        b = baseline.get(z, 0.0)
        enr = (entry["fraction"] / b) if b > 0 else None
        out.append({**entry,
                    "baseline_fraction": round(b, 5),
                    "enrichment": (round(enr, 2) if enr is not None else None)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--round0-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--finetune-steps", type=int, default=500)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--post-num-generate", type=int, default=500)
    ap.add_argument("--post-gen-batch", type=int, default=256)
    ap.add_argument("--post-max-relax", type=int, default=200)
    ap.add_argument("--lbfgs-steps", type=int, default=100)
    ap.add_argument("--force-ml-thresh", type=float, default=0.05)
    ap.add_argument("--force-strict-thresh", type=float, default=1e-4)
    ap.add_argument("--exclude-elements", type=int, nargs="*", default=[82])
    ap.add_argument("--require-oxygen", action="store_true")
    ap.add_argument("--min-vpa", type=float, default=0.0)
    ap.add_argument("--max-vpa", type=float, default=500.0)
    ap.add_argument("--min-distance", type=float, default=0.8)
    ap.add_argument("--target-vpa", type=float, default=21.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=11)
    # pass-criteria knobs
    ap.add_argument("--delta-thresh", type=float, default=0.05,
                    help="required drop in median(delta_e) eV/atom (post < pre - thresh)")
    ap.add_argument("--conv-tol", type=float, default=0.10,
                    help="max allowed drop in fraction(converged_ml) post vs pre")
    ap.add_argument("--max-memo", type=float, default=0.50,
                    help="max allowed fraction of post-valid formulas in fine-tune-seen set")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    # ---- 1. Match selected -> relaxed (use relaxed coords for fine-tune) ----
    r0 = Path(args.round0_dir)
    selected_recs = _load_jsonl(r0 / "selected.jsonl")
    relaxed_recs = _load_jsonl(r0 / "relaxed.jsonl")
    relax_index: dict[tuple, dict] = {}
    for rr in relaxed_recs:
        if rr["status"] != "ok" or not rr.get("relaxed"):
            continue
        relax_index[_match_key(rr["orig"]["z"], rr["orig"]["frac"], rr["orig"]["lattice"])] = rr

    ft_crystals: list[Crystal] = []
    seen_formulas: set[str] = set()
    n_unmatched = 0
    for sr in selected_recs:
        k = _match_key(sr["z"], sr["frac"], sr["lattice"])
        if k in relax_index:
            rr = relax_index[k]
            ft_crystals.append(Crystal(
                atom_types=torch.tensor(sr["z"], dtype=torch.long),
                frac_coords=torch.tensor(rr["relaxed"]["frac"], dtype=torch.float),
                lattice=torch.tensor(rr["relaxed"]["lattice"], dtype=torch.float),
            ))
            seen_formulas.add(reduced_formula(sr["z"]))
        else:
            n_unmatched += 1
    print(f"matched {len(ft_crystals)}/{len(selected_recs)} selected to relaxed; "
          f"{len(seen_formulas)} distinct formulas; unmatched={n_unmatched}",
          flush=True)
    if not ft_crystals:
        raise RuntimeError("no selected -> relaxed matches; aborting")

    # ---- 2. PRE distribution (round-0 relaxed.jsonl) ----
    pre_ok = [r for r in relaxed_recs if r["status"] == "ok"]
    pre_delta = [r["delta_e"] for r in pre_ok
                 if r.get("delta_e") is not None and math.isfinite(r["delta_e"])]
    pre_conv = (sum(1 for r in pre_ok if r["converged_ml"]) / len(pre_ok)) if pre_ok else 0.0
    pre_stats = {
        "n_relaxed": len(relaxed_recs),
        "n_ok": len(pre_ok),
        "fraction_converged_ml": round(pre_conv, 4),
        "delta_e_median": round(_median(pre_delta), 4) if pre_delta else None,
        "delta_e_n": len(pre_delta),
    }
    print(f"PRE: {pre_stats}", flush=True)

    # ---- 3. Fine-tune a copy of the generator on RELAXED selected ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gen = CrystalGenerator(ck["cfg"], device=args.device)
    gen.load_state_dict(ck["state_dict"])
    gen.train()
    opt = torch.optim.Adam(gen.parameters(), lr=args.finetune_lr)
    print(f"fine-tuning {args.finetune_steps} steps on {len(ft_crystals)} relaxed crystals "
          f"(lr={args.finetune_lr}) ...", flush=True)
    t0 = time.time()
    for step in range(args.finetune_steps):
        out = gen.train_step(ft_crystals)
        opt.zero_grad()
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt.step()
        if step % max(args.finetune_steps // 5, 1) == 0 or step == args.finetune_steps - 1:
            print(f"  step {step}/{args.finetune_steps}  loss={float(out['total']):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finetuned_path = out_dir / "finetuned.ckpt"
    torch.save({
        "cfg": gen.cfg, "state_dict": gen.state_dict(),
        "source_ckpt": args.ckpt,
        "finetune_steps": args.finetune_steps,
        "finetuned_on_relaxed": True,
        "round0_dir": str(r0),
        "warning": "Stage-0 score-movement test; not a validated discovery model.",
    }, finetuned_path)
    print(f"saved {finetuned_path}", flush=True)

    # ---- 4. Safety eval (Entry-8 quick-eval thresholds) ----
    print("safety eval: debug_eval_quick on finetuned ckpt ...", flush=True)
    safety_out = out_dir / "safety_eval.json"
    safety_log = out_dir / "safety_eval.log"
    p = subprocess.run([
        sys.executable, "-m", "invdesflow_al.scripts.debug_eval_quick",
        "--ckpt", str(finetuned_path),
        "--manifest", args.manifest,
        "--n-sample", "512", "--gen-batch", "256",
        "--device", args.device,
        "--out", str(safety_out),
    ], capture_output=True, text=True)
    safety_log.write_text((p.stdout or "") + "\n" + (p.stderr or ""))
    safety = {}
    for line in (p.stdout or "").splitlines()[::-1]:
        if line.startswith("RESULT_JSON "):
            try:
                safety = json.loads(line[len("RESULT_JSON "):])
            except Exception:
                safety = {}
            break
    print(f"safety: {safety}", flush=True)

    # ---- 5. POST: fresh generate + validity + pre-novelty + relax ----
    gen.eval()
    print(f"loading manifest novelty set from {args.manifest} ...", flush=True)
    train_keys = manifest_novelty_set(args.manifest)
    print(f"  {len(train_keys)} keys", flush=True)
    hist = atom_count_hist(args.manifest)

    class _V:  # mimic argparse for validity()
        pass

    val_args = _V()
    val_args.exclude_elements = args.exclude_elements
    val_args.require_oxygen = args.require_oxygen
    val_args.min_vpa = args.min_vpa
    val_args.max_vpa = args.max_vpa
    val_args.min_distance = args.min_distance

    post_gen_records: list[dict] = []
    post_valid: list[tuple] = []
    post_formulas: Counter = Counter()
    filter_counts: Counter = Counter()
    print(f"generating {args.post_num_generate} from finetuned ckpt ...", flush=True)
    t_gen = time.time()
    while len(post_gen_records) < args.post_num_generate:
        bs = min(args.post_gen_batch, args.post_num_generate - len(post_gen_records))
        idx = torch.randint(0, len(hist), (bs,), generator=rng)
        n_atoms = hist[idx].tolist()
        for c in gen.sample(n_atoms):
            zlist = c.atom_types.tolist()
            formula = reduced_formula(zlist)
            post_formulas[formula] += 1
            ok, reason, diag = validity(c, val_args)
            filter_counts[reason] += 1
            post_gen_records.append({"z": zlist, "ok": ok, "reason": reason, "formula": formula})
            if ok:
                post_valid.append((c, formula, element_set_key(zlist), diag))
        if len(post_gen_records) % max(args.post_gen_batch, 200) < args.post_gen_batch:
            print(f"  post-gen {len(post_gen_records)}/{args.post_num_generate} "
                  f"valid={len(post_valid)} ({time.time()-t_gen:.0f}s)", flush=True)

    pool = []
    for c, formula, eset, diag in post_valid:
        k = novelty_key(c)
        pool.append((k not in train_keys, c, formula, eset, k))
    pool.sort(key=lambda x: not x[0])     # novel first
    pool = pool[: args.post_max_relax]
    print(f"relaxing {len(pool)} post-finetune candidates (novel_pre first) ...", flush=True)

    cache_path = out_dir / "relax_cache.json"
    oracle = CHGNetOracle(
        cache_path=str(cache_path),
        device=args.device,
        ml_thresh=args.force_ml_thresh,
        strict_thresh=args.force_strict_thresh,
        steps=args.lbfgs_steps,
    )

    post_rows: list[dict] = []
    n_ok = n_fail = n_conv = 0
    t_rel = time.time()
    for i, (novel_pre, c, formula, eset, k) in enumerate(pool):
        r = oracle.relax_one(c)
        if r.status == "ok":
            n_ok += 1
            if r.converged_ml:
                n_conv += 1
        else:
            n_fail += 1
        post_rows.append({
            "formula": formula, "element_set": eset,
            "novel_pre": bool(novel_pre),
            "delta_e": r.delta_e, "max_force": r.max_force,
            "converged_ml": r.converged_ml, "converged_strict": r.converged_strict,
            "spacegroup_post": r.spacegroup_post,
            "status": r.status, "reason": r.reason, "cached": r.cached,
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(pool):
            print(f"  post-relax {i+1}/{len(pool)}  ok={n_ok} failed={n_fail} "
                  f"conv_ml={n_conv}  ({time.time()-t_rel:.0f}s)", flush=True)
            oracle.flush()
    oracle.flush()
    with open(out_dir / "post_relaxed.jsonl", "w") as f:
        for row in post_rows:
            f.write(json.dumps(row) + "\n")

    post_delta = [r["delta_e"] for r in post_rows
                  if r["status"] == "ok" and r.get("delta_e") is not None
                  and math.isfinite(r["delta_e"])]
    post_conv_frac = (n_conv / n_ok) if n_ok else 0.0
    post_stats = {
        "n_relaxed": len(post_rows),
        "n_ok": n_ok, "n_failed": n_fail,
        "fraction_converged_ml": round(post_conv_frac, 4),
        "delta_e_median": round(_median(post_delta), 4) if post_delta else None,
        "delta_e_n": len(post_delta),
    }
    print(f"POST: {post_stats}", flush=True)

    # ---- 6. Memorization rate (over post-valid formulas) ----
    post_valid_formulas = [f for _, f, _, _ in post_valid]
    n_memo = sum(1 for f in post_valid_formulas if f in seen_formulas)
    memo_rate = (n_memo / len(post_valid_formulas)) if post_valid_formulas else 0.0

    # ---- 7. Compare + verdict ----
    drops_dE = (
        pre_stats["delta_e_median"] - post_stats["delta_e_median"]
        if pre_stats["delta_e_median"] is not None and post_stats["delta_e_median"] is not None
        else None
    )
    conv_change = post_stats["fraction_converged_ml"] - pre_stats["fraction_converged_ml"]

    safety_pass = bool(
        safety
        and safety.get("unique_rate", 0) >= 0.5
        and safety.get("sane_fraction", 0) >= 0.95
        and 5 <= safety.get("vpa_median", 0) <= 100
        and safety.get("first_sat_t") is None
        and safety.get("nan", 1) == 0
    )
    delta_pass = bool(drops_dE is not None and drops_dE >= args.delta_thresh)
    conv_pass = bool(conv_change >= -args.conv_tol)
    memo_pass = bool(memo_rate <= args.max_memo)
    verdict = "PASS" if (safety_pass and delta_pass and conv_pass and memo_pass) else "FAIL"

    # ---- 8. Element histograms + enrichment (the Entry-17 Au follow-up) ----
    baseline = _baseline_element_fractions(
        args.manifest, Path("data_raw/element_baseline.json"))
    selected_zs = [sr["z"] for sr in selected_recs]              # round-0 selected
    post_valid_zs = [c.atom_types.tolist() for c, _, _, _ in post_valid]
    hist_selected = _element_histogram(selected_zs, top_n=15)
    hist_post_valid = _element_histogram(post_valid_zs, top_n=15)
    enr_selected = _enrichment(hist_selected, baseline)
    enr_post_valid = _enrichment(hist_post_valid, baseline)
    print()
    print("Element enrichment (post-finetune valid; top 10 by post fraction):")
    print(f"  {'Z':>4} {'count':>6} {'post_frac':>10} {'baseline':>10} {'enrich':>8}")
    for e in enr_post_valid[:10]:
        enr = e["enrichment"]
        enr_str = "inf" if enr is None else f"{enr:6.2f}"
        print(f"  {e['z']:>4} {e['count']:>6} {e['fraction']:>10.4f} "
              f"{e['baseline_fraction']:>10.4f} {enr_str:>8}")

    compare = {
        "source_ckpt": args.ckpt,
        "round0_dir": str(r0),
        "finetuned_ckpt": str(finetuned_path),
        "finetune_steps": args.finetune_steps,
        "finetune_lr": args.finetune_lr,
        "n_finetune_crystals": len(ft_crystals),
        "finetune_seen_formulas": len(seen_formulas),
        "pre": pre_stats,
        "post": post_stats,
        "post_gen_summary": {
            "generated": len(post_gen_records),
            "valid": len(post_valid),
            "valid_fraction": round(len(post_valid) / max(len(post_gen_records), 1), 4),
            "filter_counts": dict(filter_counts),
            "distinct_formulas_generated": len(post_formulas),
        },
        "memorization": {
            "n_post_valid": len(post_valid_formulas),
            "n_in_finetune_seen": n_memo,
            "rate": round(memo_rate, 4),
        },
        "safety_eval": safety,
        "thresholds": {
            "delta_thresh": args.delta_thresh,
            "conv_tol": args.conv_tol,
            "max_memo": args.max_memo,
        },
        "gates": {
            "safety_pass": safety_pass,
            "delta_pass": delta_pass,
            "conv_pass": conv_pass,
            "memo_pass": memo_pass,
            "delta_drop_eV_per_atom": round(drops_dE, 4) if drops_dE is not None else None,
            "conv_change": round(conv_change, 4),
        },
        "element_distribution": {
            "selected_round0": hist_selected,
            "post_finetune_valid": hist_post_valid,
            "selected_round0_enrichment_top": enr_selected,
            "post_finetune_valid_enrichment_top": enr_post_valid,
            "max_element_fraction_post": (
                max((e["fraction"] for e in hist_post_valid["top"]), default=0.0)
            ),
            "max_enrichment_post": max(
                (e["enrichment"] for e in enr_post_valid if e["enrichment"] is not None),
                default=None,
            ),
        },
        "verdict": verdict,
    }
    with open(out_dir / "compare.json", "w") as f:
        json.dump(compare, f, indent=2)
    print()
    print(f"PRE  median(delta_e) = {pre_stats['delta_e_median']}  "
          f"fraction_conv_ml = {pre_stats['fraction_converged_ml']}")
    print(f"POST median(delta_e) = {post_stats['delta_e_median']}  "
          f"fraction_conv_ml = {post_stats['fraction_converged_ml']}")
    print(f"drop_dE = {compare['gates']['delta_drop_eV_per_atom']} "
          f"(thresh >= {args.delta_thresh})   "
          f"conv_change = {compare['gates']['conv_change']} (tol >= -{args.conv_tol})   "
          f"memo_rate = {memo_rate:.3f} (max {args.max_memo})")
    print(f"gates: safety={safety_pass} delta={delta_pass} conv={conv_pass} memo={memo_pass}")
    print(f"VERDICT: {verdict}")
    print("RESULT_JSON " + json.dumps(compare))


if __name__ == "__main__":
    main()
