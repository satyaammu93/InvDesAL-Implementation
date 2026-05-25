"""Tiny active-learning dry run.

This script is a pipeline/debugging harness, not a discovery workflow. It uses
the validated generator to:

  1. generate a small candidate pool;
  2. apply transparent structural/composition validity filters;
  3. rank candidates with a deliberately simple heuristic proxy;
  4. select a diverse top-k set;
  5. optionally fine-tune a COPY of the generator on that selected set.

The choices are intentionally conservative and documented in the JSON summary:
the score is a placeholder until a real stability/property oracle is wired in
(FormEGNN/DPA-2/CHGNet/M3GNet or a piezoelectric model).

Example:
  python -m invdesflow_al.scripts.run_tiny_al_dryrun \
      --ckpt gen_10k_ax0.ckpt --manifest data_raw/pretrain_10k.jsonl \
      --num-generate 1000 --top-k 50 --device cuda \
      --out-dir al_runs/tiny_stability_dryrun
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import torch

from ..data.datasets import StructureRecord, load_structures
from ..models.generator import CrystalGenerator


def reduced_formula(z: list[int]) -> str:
    cnt = Counter(z)
    g = reduce(gcd, cnt.values()) if cnt else 1
    return "-".join(f"{el}{n // g}" for el, n in sorted(cnt.items()))


def element_set_key(z: list[int]) -> str:
    return "-".join(map(str, sorted(set(z))))


def lattice_vpa(c) -> float:
    return float(torch.det(c.lattice).abs()) / max(c.num_atoms, 1)


def min_periodic_distance(c) -> float:
    """Minimum pair distance using a fractional minimum-image convention.

    This is a cheap structural sanity check. It is not a relaxation and it does
    not know covalent/ionic radii; it only catches obvious atom overlaps.
    """
    n = c.num_atoms
    if n < 2:
        return float("inf")
    f = c.frac_coords
    d = f[:, None, :] - f[None, :, :]
    d = d - d.round()
    cart = torch.einsum("...i,ij->...j", d, c.lattice)
    dist = cart.norm(dim=-1)
    dist[torch.eye(n, dtype=torch.bool)] = float("inf")
    return float(dist.min())


def structure_record(c, *, score: float | None = None, reason: str = "") -> StructureRecord:
    meta = {}
    if score is not None:
        meta["dryrun_score"] = float(score)
    if reason:
        meta["filter_reason"] = reason
    return StructureRecord.from_crystal(c, source="tiny_al_dryrun", meta=meta)


def validity(c, args) -> tuple[bool, str, dict]:
    """Return validity, reason, and cheap diagnostics for one candidate."""
    if not torch.isfinite(c.lattice).all() or not torch.isfinite(c.frac_coords).all():
        return False, "nan_or_inf", {}
    det = float(torch.det(c.lattice))
    if not math.isfinite(det) or abs(det) <= 1e-8:
        return False, "degenerate_lattice", {"det": det}
    vpa = abs(det) / max(c.num_atoms, 1)
    if vpa <= args.min_vpa or vpa > args.max_vpa:
        return False, "volume_per_atom", {"vpa": vpa}
    z = c.atom_types.tolist()
    banned = set(args.exclude_elements)
    if banned and banned.intersection(z):
        return False, "excluded_element", {"excluded": sorted(banned.intersection(z))}
    if args.require_oxygen and 8 not in z:
        return False, "requires_oxygen", {}
    dmin = min_periodic_distance(c)
    if dmin < args.min_distance:
        return False, "atom_overlap", {"min_distance": dmin, "vpa": vpa}
    return True, "ok", {"vpa": vpa, "min_distance": dmin}


def dryrun_score(c, diag: dict, args) -> float:
    """Transparent placeholder score for pipeline testing.

    The proxy rewards:
      * volume/atom near a configurable target (default 21 A^3/atom, from the
        pretrain data median observed in the gates);
      * non-overlapping atoms with a bit of margin;
      * oxygen presence if requested/desired for ceramic dry runs.

    This is intentionally not a materials-property prediction. Its only job is
    to produce a deterministic ranking so selection/fine-tuning plumbing can be
    tested before a real oracle exists.
    """
    vpa = max(float(diag.get("vpa", lattice_vpa(c))), 1e-8)
    dmin = float(diag.get("min_distance", min_periodic_distance(c)))
    vpa_term = -abs(math.log(vpa / args.target_vpa))
    dist_term = min(dmin, 3.0) / 3.0
    oxygen_term = 0.25 if 8 in c.atom_types.tolist() else 0.0
    # Mildly prefer chemically richer candidates without forcing high entropy.
    elem_term = min(len(set(c.atom_types.tolist())), 5) / 20.0
    return vpa_term + dist_term + oxygen_term + elem_term


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    t = torch.tensor(xs, dtype=torch.float).sort().values
    pick = lambda q: float(t[min(len(t) - 1, int(q * len(t)))])
    return {
        "min": round(float(t[0]), 4),
        "p5": round(pick(0.05), 4),
        "median": round(float(t[len(t) // 2]), 4),
        "p95": round(pick(0.95), 4),
        "max": round(float(t[-1]), 4),
    }


def write_jsonl(path: Path, rows: list[StructureRecord]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(r.to_json() + "\n")


def load_generator(path: str, device: str) -> tuple[CrystalGenerator, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    gen = CrystalGenerator(ckpt["cfg"], device=device)
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()
    return gen, ckpt


def atom_count_hist(manifest: str, cap: int = 200000) -> torch.Tensor:
    counts = []
    for i, r in enumerate(load_structures(manifest)):
        counts.append(len(r.z))
        if i + 1 >= cap:
            break
    if not counts:
        raise ValueError(f"no atom counts found in {manifest}")
    return torch.tensor(counts, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="al_runs/tiny_dryrun")
    ap.add_argument("--num-generate", type=int, default=1000)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-elements", type=int, nargs="*", default=[82],
                    help="atomic numbers to ban; default excludes Pb for later lead-free work")
    ap.add_argument("--require-oxygen", action="store_true",
                    help="optional ceramic/oxide dry-run filter")
    ap.add_argument("--min-vpa", type=float, default=0.0)
    ap.add_argument("--max-vpa", type=float, default=500.0)
    ap.add_argument("--min-distance", type=float, default=0.8)
    ap.add_argument("--target-vpa", type=float, default=21.0)
    ap.add_argument("--finetune-steps", type=int, default=0,
                    help="optional plumbing test: fine-tune a copy on selected candidates")
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--finetune-ckpt", default="",
                    help="where to save optional fine-tuned copy")
    # Entry-14 Stage-0 oracle additions
    ap.add_argument("--oracle", choices=["heuristic", "chgnet"], default="heuristic",
                    help="placeholder score (heuristic) vs CHGNet relaxation oracle "
                         "(Stage 0: score = delta_e * I_relax_ml * I_novelty_pre)")
    ap.add_argument("--oracle-max-candidates", type=int, default=200,
                    help="cap on candidates sent to the oracle per round")
    ap.add_argument("--force-converged-ml-thresh", type=float, default=0.05,
                    help="ML practitioner threshold for max_force (eV/A) -- round-0 gate")
    ap.add_argument("--force-converged-strict-thresh", type=float, default=1e-4,
                    help="paper threshold for max_force (eV/A) -- recorded only")
    ap.add_argument("--lbfgs-steps", type=int, default=100,
                    help="max LBFGS steps per CHGNet relaxation")
    ap.add_argument("--no-relax-cache", action="store_true",
                    help="disable the persistent relax_cache.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    gen, ckpt = load_generator(args.ckpt, args.device)
    hist = atom_count_hist(args.manifest)

    generated = []
    valid = []
    selected = []
    filter_counts = Counter()
    formulas = Counter()
    valid_formulas = Counter()
    vpas = []
    dmins = []
    t0 = time.time()

    while len(generated) < args.num_generate:
        bs = min(args.gen_batch, args.num_generate - len(generated))
        idx = torch.randint(0, len(hist), (bs,), generator=rng)
        n_atoms = hist[idx].tolist()
        for c in gen.sample(n_atoms):
            formula = reduced_formula(c.atom_types.tolist())
            formulas[formula] += 1
            ok, reason, diag = validity(c, args)
            filter_counts[reason] += 1
            generated.append(structure_record(c, reason=reason))
            if ok:
                score = dryrun_score(c, diag, args)
                rec = structure_record(c, score=score, reason="ok")
                rec.meta.update({
                    "formula": formula,
                    "element_set": element_set_key(c.atom_types.tolist()),
                    "vpa": float(diag["vpa"]),
                    "min_distance": float(diag["min_distance"]),
                })
                valid.append((score, c, rec))
                valid_formulas[formula] += 1
                vpas.append(float(diag["vpa"]))
                dmins.append(float(diag["min_distance"]))
        if len(generated) % max(args.gen_batch, 500) < args.gen_batch:
            print(f"generated {len(generated)}/{args.num_generate} "
                  f"valid={len(valid)} ({time.time()-t0:.0f}s)", flush=True)

    # --- Entry-14 Stage-0 oracle (optional) ---------------------------------
    # If --oracle chgnet, relax up to --oracle-max-candidates of the validity
    # survivors via CHGNet, compute Stage-0 score (delta_e * I_relax_ml *
    # I_novelty_pre), and select from the relaxed+converged_ml+novel_post set.
    # Else: keep the heuristic path unchanged.
    oracle_summary: dict = {}
    relaxed_rows: list[dict] = []
    if args.oracle == "chgnet":
        from ..al import CHGNetOracle, manifest_novelty_set, novelty_key

        print(f"[chgnet] building manifest novelty set from {args.manifest} ...", flush=True)
        train_keys = manifest_novelty_set(args.manifest)
        print(f"[chgnet]   {len(train_keys)} unique (formula, spacegroup) keys", flush=True)

        cache_path = None if args.no_relax_cache else str(out_dir / "relax_cache.json")
        oracle = CHGNetOracle(
            cache_path=cache_path,
            device=args.device,
            ml_thresh=args.force_converged_ml_thresh,
            strict_thresh=args.force_converged_strict_thresh,
            steps=args.lbfgs_steps,
        )

        # pre-relax novelty + cap. Novel candidates first, then by heuristic score.
        pool = []
        for score, c, rec in valid:
            k = novelty_key(c)
            novel_pre = k not in train_keys
            rec.meta["novel_pre"] = novel_pre
            rec.meta["novelty_key_pre"] = list(k) if k[0] is not None else None
            pool.append((score, c, rec, k, novel_pre))
        pool.sort(key=lambda t: (not t[4], -t[0]))
        cand = pool[: args.oracle_max_candidates]
        print(f"[chgnet] relaxing {len(cand)} candidates (cap "
              f"{args.oracle_max_candidates}, novel_pre first) ...", flush=True)

        n_ok = n_failed = n_conv_ml = n_conv_strict = 0
        t_rel = time.time()
        for i, (h_score, c, rec, k, novel_pre) in enumerate(cand):
            r = oracle.relax_one(c)
            if r.status == "ok":
                n_ok += 1
                if r.converged_ml:
                    n_conv_ml += 1
                if r.converged_strict:
                    n_conv_strict += 1
                post_key = (k[0], r.spacegroup_post)
                novel_post = post_key not in train_keys
                stage0_score = (
                    r.delta_e
                    * (1.0 if r.converged_ml else 0.0)
                    * (1.0 if novel_pre else 0.0)
                )
            else:
                n_failed += 1
                novel_post = False
                stage0_score = float("-inf")
            rec.meta.update({
                "novel_post": bool(novel_post),
                "stage": 0,
                "delta_e": r.delta_e,
                "max_force": r.max_force,
                "converged_ml": r.converged_ml,
                "converged_strict": r.converged_strict,
                "volume_change": r.volume_change,
                "min_distance_post": r.min_distance_post,
                "spacegroup_pre": r.spacegroup_pre,
                "spacegroup_post": r.spacegroup_post,
                "status": r.status,
                "reason": r.reason,
                "stage0_score": (None if not math.isfinite(stage0_score)
                                 else float(stage0_score)),
            })
            relaxed_rows.append({
                "orig": {
                    "z": c.atom_types.tolist(),
                    "frac": [list(map(float, fr)) for fr in c.frac_coords.tolist()],
                    "lattice": [list(map(float, row)) for row in c.lattice.tolist()],
                },
                "relaxed": ({"frac": r.relaxed_frac, "lattice": r.relaxed_lattice}
                            if r.status == "ok" else None),
                "energy_per_atom": r.energy_per_atom,
                "max_force": r.max_force,
                "converged_ml": r.converged_ml,
                "converged_strict": r.converged_strict,
                "delta_e": r.delta_e,
                "volume_change": r.volume_change,
                "min_distance_post": r.min_distance_post,
                "spacegroup_pre": r.spacegroup_pre,
                "spacegroup_post": r.spacegroup_post,
                "novel_pre": bool(novel_pre),
                "novel_post": bool(novel_post),
                "e_form": None,
                "stage": 0,
                "score": (None if not math.isfinite(stage0_score)
                          else float(stage0_score)),
                "status": r.status, "reason": r.reason, "cached": r.cached,
                "formula": rec.meta.get("formula"),
                "element_set": rec.meta.get("element_set"),
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(cand):
                print(f"[chgnet]   {i+1}/{len(cand)}  ok={n_ok} failed={n_failed} "
                      f"conv_ml={n_conv_ml} conv_strict={n_conv_strict}  "
                      f"({time.time()-t_rel:.0f}s)", flush=True)
                oracle.flush()
        oracle.flush()

        # write relaxed.jsonl
        with open(out_dir / "relaxed.jsonl", "w") as f:
            for row in relaxed_rows:
                f.write(json.dumps(row) + "\n")

        # selection from converged + post-novel set, diverse
        eligible = [
            (rr["score"], rec, rr)
            for (rr, (_, _, rec, _, _)) in zip(relaxed_rows, cand)
            if rr["status"] == "ok" and rr["converged_ml"] and rr["novel_post"]
            and rr["score"] is not None
        ]
        eligible.sort(key=lambda x: -x[0])
        element_set_counts = Counter()
        seen_formula = set()
        for score, rec, rr in eligible:
            formula = rec.meta["formula"]
            eset = rec.meta["element_set"]
            if formula in seen_formula:
                continue
            if element_set_counts[eset] >= 3:
                continue
            selected.append(rec)
            seen_formula.add(formula)
            element_set_counts[eset] += 1
            if len(selected) >= args.top_k:
                break

        oracle_summary = {
            "backend": "chgnet",
            "cache_path": cache_path,
            "manifest_novelty_keys": len(train_keys),
            "candidates_pre_cap": len(pool),
            "relaxed": len(cand),
            "ok": n_ok,
            "failed": n_failed,
            "converged_ml": n_conv_ml,
            "converged_strict": n_conv_strict,
            "ok_fraction": round(n_ok / max(len(cand), 1), 4),
            "converged_ml_fraction_of_ok": round(n_conv_ml / max(n_ok, 1), 4),
            "converged_strict_fraction_of_ok": round(n_conv_strict / max(n_ok, 1), 4),
            "eligible_after_post_novelty": len(eligible),
            "selected": len(selected),
            "ml_thresh_ev_per_A": args.force_converged_ml_thresh,
            "strict_thresh_ev_per_A": args.force_converged_strict_thresh,
            "stage": 0,
        }
    else:
        # heuristic path (existing): diverse selection from `valid`
        element_set_counts = Counter()
        seen_formula = set()
        for score, c, rec in sorted(valid, key=lambda x: x[0], reverse=True):
            formula = rec.meta["formula"]
            eset = rec.meta["element_set"]
            if formula in seen_formula:
                continue
            if element_set_counts[eset] >= 3:
                continue
            selected.append(rec)
            seen_formula.add(formula)
            element_set_counts[eset] += 1
            if len(selected) >= args.top_k:
                break

    write_jsonl(out_dir / "generated.jsonl", generated)
    write_jsonl(out_dir / "valid.jsonl", [r for _, _, r in valid])
    write_jsonl(out_dir / "selected.jsonl", selected)

    finetune_path = None
    if args.finetune_steps > 0 and selected:
        # This is only a mechanics test: there are no new property labels yet,
        # so we fine-tune on the selected generated structures as pseudo-data.
        gen.train()
        opt = torch.optim.Adam(gen.parameters(), lr=args.finetune_lr)
        crystals = [r.to_crystal() for r in selected]
        for step in range(args.finetune_steps):
            out = gen.train_step(crystals)
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            opt.step()
            if step % max(args.finetune_steps // 5, 1) == 0:
                print(f"finetune step {step}/{args.finetune_steps} "
                      f"loss={float(out['total']):.4f}", flush=True)
        finetune_path = args.finetune_ckpt or str(out_dir / "tiny_al_finetuned.ckpt")
        torch.save({
            "cfg": gen.cfg,
            "state_dict": gen.state_dict(),
            "source_ckpt": args.ckpt,
            "dryrun_selected_jsonl": str(out_dir / "selected.jsonl"),
            "dryrun_warning": (
                "Fine-tuned on heuristic-selected generated structures only; "
                "not a validated discovery model."
            ),
        }, finetune_path)

    def _score_for(r) -> float:
        if "stage0_score" in r.meta and r.meta["stage0_score"] is not None:
            return float(r.meta["stage0_score"])
        return float(r.meta.get("dryrun_score", float("nan")))

    score_label = ("Stage-0 CHGNet: S = delta_e * I_relax_ml * I_novelty_pre"
                   if args.oracle == "chgnet" else
                   "heuristic: volume/atom near target_vpa + min-distance margin"
                   " + oxygen bonus + mild element-diversity term")

    summary = {
        "purpose": "tiny active-learning dry run for plumbing/debugging, not discovery",
        "source_ckpt": args.ckpt,
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_loss": ckpt.get("val_loss"),
        "manifest": args.manifest,
        "oracle": args.oracle,
        "choices": {
            "validator": {
                "exclude_elements": args.exclude_elements,
                "require_oxygen": args.require_oxygen,
                "min_vpa": args.min_vpa,
                "max_vpa": args.max_vpa,
                "min_distance": args.min_distance,
            },
            "score": score_label,
            "target_vpa": args.target_vpa,
            "selection": ("score sorted on (delta_e * gates); one per formula; "
                          "max 3 per element set; converged_ml + novel_post required"
                          if args.oracle == "chgnet" else
                          "score sorted; one per formula; max 3 per element set"),
            "real_oracle_missing": (
                "Stage 0: stability proxy only (CHGNet ΔE). Not a piezoelectric"
                " or formation-energy oracle yet — Stages 1–3 in PROGRESS Entry 14."
                if args.oracle == "chgnet" else
                "Replace heuristic score with relaxation/stability/property "
                "oracle before trusting candidates."
            ),
        },
        "generated": len(generated),
        "valid": len(valid),
        "selected": len(selected),
        "valid_fraction": round(len(valid) / max(len(generated), 1), 4),
        "distinct_formulas_generated": len(formulas),
        "distinct_formulas_valid": len(valid_formulas),
        "filter_counts": dict(filter_counts),
        "vpa_valid": quantiles(vpas),
        "min_distance_valid": quantiles(dmins),
        "oracle_summary": oracle_summary,
        "top_selected": [
            {
                "score": round(_score_for(r), 4),
                "formula": r.meta["formula"],
                "element_set": r.meta["element_set"],
                "vpa": round(float(r.meta["vpa"]), 4),
                "min_distance": round(float(r.meta["min_distance"]), 4),
                **({"delta_e": round(float(r.meta["delta_e"]), 4),
                    "max_force": round(float(r.meta["max_force"]), 4),
                    "converged_ml": bool(r.meta["converged_ml"]),
                    "converged_strict": bool(r.meta["converged_strict"]),
                    "spacegroup_post": r.meta.get("spacegroup_post"),
                    "novel_post": bool(r.meta.get("novel_post", False))}
                   if "delta_e" in r.meta else {}),
            }
            for r in selected[:10]
        ],
        "finetuned_ckpt": finetune_path,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
