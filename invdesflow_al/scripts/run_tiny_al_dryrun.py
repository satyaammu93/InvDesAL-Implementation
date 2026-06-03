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


def is_centrosymmetric_crystal(c, symprec: float = 0.1) -> bool | None:
    """Return True when pymatgen finds an inversion symmetry operation."""
    try:
        import numpy as np
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        lat = c.lattice.detach().cpu().numpy() if hasattr(c.lattice, "detach") else c.lattice
        frac = (c.frac_coords.detach().cpu().numpy()
                if hasattr(c.frac_coords, "detach") else c.frac_coords)
        z = c.atom_types.tolist() if hasattr(c.atom_types, "tolist") else list(c.atom_types)
        s = Structure(Lattice(lat), species=z, coords=frac, coords_are_cartesian=False)
        ops = SpacegroupAnalyzer(s, symprec=symprec).get_symmetry_operations(
            cartesian=False)
        inv = -np.eye(3)
        return any(np.allclose(op.rotation_matrix, inv, atol=1e-6) for op in ops)
    except Exception:
        return None


def is_centrosymmetric_relaxed(z, frac, lattice, symprec: float = 0.1) -> bool | None:
    c = type("_CrystalLike", (), {})()
    c.atom_types = list(z)
    c.frac_coords = frac
    c.lattice = lattice
    return is_centrosymmetric_crystal(c, symprec=symprec)


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
    if getattr(args, "reject_centrosymmetric", False):
        is_centro = is_centrosymmetric_crystal(c)
        if is_centro is None:
            return False, "symmetry_analysis_failed", {"vpa": vpa, "min_distance": dmin}
        if is_centro:
            return False, "centrosymmetric_pre", {"vpa": vpa, "min_distance": dmin}
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


def load_scarcity_weights(mode: str, prior_path: str | None, min_w: float) -> dict[int, float]:
    """Per-Z weight table W(z) used to suppress element-bias amplification.

    mode='none' (or no prior): empty dict; the caller treats missing Z as W=1.
    mode='inv-enrichment': W(z) = 1 / max(enrichment(z), 1.0), floored at min_w.

    `prior_path` may point to:
      (a) a compare.json with `element_distribution.post_finetune_valid_enrichment_top`
          (preferred — fresh from the previous round), or
      (b) a flat JSON dict {Z: enrichment(float)}.

    Z keys may be int or str; both are normalized to int.
    """
    if mode == "none" or not prior_path:
        return {}
    raw = json.loads(Path(prior_path).read_text())
    enrich: dict[int, float] = {}
    if isinstance(raw, dict) and "element_distribution" in raw:
        for e in raw["element_distribution"].get("post_finetune_valid_enrichment_top", []):
            if e.get("enrichment") is not None:
                enrich[int(e["z"])] = float(e["enrichment"])
    elif isinstance(raw, dict):
        for k, v in raw.items():
            try:
                enrich[int(k)] = float(v)
            except (TypeError, ValueError):
                pass
    weights = {z: max(min_w, 1.0 / max(en, 1.0)) for z, en in enrich.items()}
    return weights


def composition_weight(z_list: list[int], wtable: dict[int, float]) -> float:
    """Per-atom-averaged scarcity weight. Default W=1 for Z not in the table."""
    if not z_list:
        return 1.0
    return sum(wtable.get(int(z), 1.0) for z in z_list) / len(z_list)


# --- Entry 23: soft family prior for piezoelectric target chemistry ---------
# Symmetry alone (Plan A) admits phosphate/chromate non-centro oxides while
# losing Nb-containing perovskites. The piezo prior rewards candidates that
# contain a known piezo-active B-site cation (Nb/Ti/Fe) and, additionally,
# an alkali / alkaline-earth / Bi A-site cation (Na/K/Ba/Bi). It is a soft
# multiplicative bonus on the score — never a hard gate — so diversity is
# preserved and the AL signal can still find unexpected chemistries.
PIEZO_B_SITE = (22, 26, 41)     # Ti, Fe, Nb
PIEZO_A_SITE = (11, 19, 56, 83)  # Na, K, Ba, Bi


def compute_family_bonus(
    z_list: list[int],
    mode: str,
    b_bonus: float,
    ab_bonus: float,
) -> float:
    """Multiplicative score bonus for piezo-family compositions.

    mode='none' -> 1.0 (disabled).
    mode='piezo':
       1.0                              if no Ti/Fe/Nb
       1.0 + b_bonus                    if any of Ti/Fe/Nb (B-site present)
       1.0 + b_bonus + ab_bonus         if also any of Na/K/Ba/Bi (A-site present)

    With defaults b_bonus=0.5, ab_bonus=0.5: bonus is 1.0 / 1.5 / 2.0.
    """
    if mode == "none" or not z_list:
        return 1.0
    has_B = any(int(z) in PIEZO_B_SITE for z in z_list)
    has_A = any(int(z) in PIEZO_A_SITE for z in z_list)
    bonus = 1.0
    if has_B:
        bonus += b_bonus
        if has_A:
            bonus += ab_bonus
    return bonus


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
    ap.add_argument("--reject-centrosymmetric", action="store_true",
                    help="reject generated candidates whose pre-relaxation symmetry "
                         "contains inversion")
    ap.add_argument("--finetune-steps", type=int, default=0,
                    help="optional plumbing test: fine-tune a copy on selected candidates")
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--finetune-ckpt", default="",
                    help="where to save optional fine-tuned copy")
    ap.add_argument("--finetune-on", choices=["original", "relaxed"], default=None,
                    help="train fine-tune on the generator's original outputs or on "
                         "the CHGNet-relaxed selected structures. Default: "
                         "'relaxed' for --oracle chgnet, 'original' otherwise.")
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
    ap.add_argument("--reject-centrosymmetric-post", action="store_true",
                    help="for CHGNet oracle runs, reject relaxed structures whose "
                         "post-relaxation symmetry contains inversion")
    # Plan C': soft composition-aware score
    ap.add_argument("--scarcity-mode", choices=["none", "inv-enrichment"], default="none",
                    help="multiply Stage-0 score by a per-atom-averaged element "
                         "weight to suppress single-element bias amplification. "
                         "inv-enrichment: W(z)=1/max(enrichment(z), 1) using "
                         "--enrichment-prior; baseline=1 for Z not in prior.")
    ap.add_argument("--enrichment-prior", default=None,
                    help="path to a prior compare.json (uses its "
                         "element_distribution.post_finetune_valid_enrichment_top) "
                         "OR a flat JSON dict {Z(int|str): enrichment(float)}.")
    ap.add_argument("--scarcity-min-weight", type=float, default=0.01,
                    help="floor for per-Z W(z) so no single atom can zero a score")
    # Entry 23: soft composition family prior (piezoelectric target chemistry)
    ap.add_argument("--family-prior", choices=["none", "piezo"], default="none",
                    help="multiplicative score bonus for piezo-target families. "
                         "piezo: +b_bonus if Ti/Fe/Nb present, +ab_bonus more if "
                         "also Na/K/Ba/Bi present. Soft — never a hard gate.")
    ap.add_argument("--family-bonus-b", type=float, default=0.5,
                    help="bonus added when a B-site cation (Ti/Fe/Nb) is present")
    ap.add_argument("--family-bonus-ab", type=float, default=0.5,
                    help="additional bonus when both a B-site and an A-site "
                         "(Na/K/Ba/Bi) cation are present")
    # Stage 1 (Entry 14 v2 / Entry 20): paper-faithful E_form
    ap.add_argument("--use-e-form", action="store_true",
                    help="Stage 1: compute E_form per paper Eq. 1 via lazy elemental "
                         "CHGNet references. Score becomes "
                         "S = (-E_form) * I_relax_ml * I_novelty_pre * W_comp. "
                         "Candidates with missing refs fall back to Stage-0 ΔE.")
    ap.add_argument("--elemental-refs-cache",
                    default="data_raw/chgnet_elemental_refs.json",
                    help="persistent JSON cache for elemental reference energies")
    # Plan C / Stage 3: piezoelectric scoring head
    ap.add_argument("--piezo-head", default=None,
                    help="path to PiezoHead checkpoint. When set, the score "
                         "is multiplied by max(predicted |e_max|, --piezo-floor): "
                         "S = (-E_form) * max(|e_max|, floor) * gates * W_comp * family.")
    ap.add_argument("--piezo-floor", type=float, default=0.05,
                    help="floor for the piezo factor in C/m^2; prevents low-piezo "
                         "candidates from being zeroed out (default 0.05, roughly the "
                         "p10 of the matminer piezoelectric_tensor dataset)")
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
        # Stage 1 (--use-e-form): lazy elemental refs for paper E_form
        refs = None
        if args.use_e_form:
            from ..al import ElementalRefs
            refs = ElementalRefs(args.elemental_refs_cache, oracle)
            print(f"[stage1] E_form enabled; refs cache "
                  f"{args.elemental_refs_cache} "
                  f"(cov: {refs.coverage()})", flush=True)
        n_e_form_ok = n_e_form_missing = 0

        # Plan C': load scarcity weights (empty dict for mode='none')
        scarcity_w = load_scarcity_weights(
            args.scarcity_mode, args.enrichment_prior, args.scarcity_min_weight)
        if scarcity_w:
            top = sorted(scarcity_w.items(), key=lambda x: x[1])[:10]
            print(f"[chgnet] scarcity-mode={args.scarcity_mode}  "
                  f"prior={args.enrichment_prior}  "
                  f"covered_Z={len(scarcity_w)}  "
                  f"most-penalized: {[(z, round(w, 3)) for z, w in top]}",
                  flush=True)
        else:
            print(f"[chgnet] scarcity-mode={args.scarcity_mode} (no penalty applied)",
                  flush=True)
        if args.family_prior != "none":
            print(f"[chgnet] family-prior={args.family_prior}  "
                  f"B-site Z={PIEZO_B_SITE} bonus=+{args.family_bonus_b}  "
                  f"A-site Z={PIEZO_A_SITE} additional=+{args.family_bonus_ab}  "
                  f"max bonus={1.0 + args.family_bonus_b + args.family_bonus_ab}",
                  flush=True)
        n_family_b_only = n_family_ab = n_family_neither = 0

        # Plan C / Stage 3: optional piezoelectric scoring head
        piezo_oracle = None
        piezo_scores: list[float] = []
        if args.piezo_head:
            from ..al import PiezoOracle
            piezo_oracle = PiezoOracle(args.piezo_head, device=args.device)
            print(f"[piezo] PiezoOracle loaded from {args.piezo_head}  "
                  f"epoch={piezo_oracle.ckpt_epoch}  "
                  f"val_rho={piezo_oracle.val_spearman:.4f}  "
                  f"floor={args.piezo_floor} C/m^2",
                  flush=True)

        print(f"[chgnet] relaxing {len(cand)} candidates (cap "
              f"{args.oracle_max_candidates}, novel_pre first) ...", flush=True)

        n_ok = n_failed = n_conv_ml = n_conv_strict = 0
        n_post_centrosymmetric = n_post_non_centrosymmetric = n_post_sym_fail = 0
        t_rel = time.time()
        for i, (h_score, c, rec, k, novel_pre) in enumerate(cand):
            r = oracle.relax_one(c)
            zlist = c.atom_types.tolist()
            comp_w = composition_weight(zlist, scarcity_w)
            family_bonus = compute_family_bonus(
                zlist, args.family_prior,
                args.family_bonus_b, args.family_bonus_ab)
            piezo_raw: float | None = None
            piezo_factor = 1.0
            # bookkeeping (only meaningful when family_prior != 'none')
            if args.family_prior != "none":
                if family_bonus >= 1.0 + args.family_bonus_b + args.family_bonus_ab - 1e-9:
                    n_family_ab += 1
                elif family_bonus >= 1.0 + args.family_bonus_b - 1e-9:
                    n_family_b_only += 1
                else:
                    n_family_neither += 1
            e_form = None; e_form_reason = None; stage_used = 0
            post_centrosymmetric = None
            post_symmetry_ok = True
            if r.status == "ok":
                n_ok += 1
                if r.converged_ml:
                    n_conv_ml += 1
                if r.converged_strict:
                    n_conv_strict += 1
                post_key = (k[0], r.spacegroup_post)
                novel_post = post_key not in train_keys
                if args.reject_centrosymmetric_post:
                    post_centrosymmetric = is_centrosymmetric_relaxed(
                        zlist, r.relaxed_frac, r.relaxed_lattice)
                    if post_centrosymmetric is None:
                        n_post_sym_fail += 1
                        post_symmetry_ok = False
                    elif post_centrosymmetric:
                        n_post_centrosymmetric += 1
                        post_symmetry_ok = False
                    else:
                        n_post_non_centrosymmetric += 1
                # Stage 1: compute E_form via lazy elemental refs; fallback to ΔE
                if refs is not None:
                    e_form, e_form_reason = refs.e_form_per_atom(zlist, r.energy_per_atom)
                    if e_form is not None:
                        n_e_form_ok += 1
                        stage_used = 1
                    else:
                        n_e_form_missing += 1
                gates = ((1.0 if r.converged_ml else 0.0)
                         * (1.0 if novel_pre else 0.0))
                if args.reject_centrosymmetric_post:
                    gates *= (1.0 if post_symmetry_ok else 0.0)
                # Plan C piezo factor (only when the candidate relaxed)
                if piezo_oracle is not None:
                    piezo_raw = piezo_oracle.score_relaxed(
                        zlist, r.relaxed_frac, r.relaxed_lattice)
                    piezo_scores.append(piezo_raw)
                    piezo_factor = max(piezo_raw, args.piezo_floor)
                if stage_used == 1:
                    # paper Eq. 1: higher S for more-negative E_form
                    stage0_score = (-e_form) * gates * comp_w * family_bonus * piezo_factor
                else:
                    stage0_score = r.delta_e * gates * comp_w * family_bonus * piezo_factor
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
                "post_centrosymmetric": post_centrosymmetric,
                "post_symmetry_ok": bool(post_symmetry_ok),
                "status": r.status,
                "reason": r.reason,
                "stage0_score": (None if not math.isfinite(stage0_score)
                                 else float(stage0_score)),
                "composition_weight": float(comp_w),
                "family_bonus": float(family_bonus),
                "piezo_e_max": (float(piezo_raw) if piezo_raw is not None else None),
                "piezo_factor": float(piezo_factor),
                "e_form": e_form,
                "e_form_reason": e_form_reason,
                "stage": stage_used,
                "relaxed_frac": r.relaxed_frac,        # carried for finetune
                "relaxed_lattice": r.relaxed_lattice,  # (large but JSONL only)
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
                "post_centrosymmetric": post_centrosymmetric,
                "post_symmetry_ok": bool(post_symmetry_ok),
                "novel_pre": bool(novel_pre),
                "novel_post": bool(novel_post),
                "e_form": e_form,
                "e_form_reason": e_form_reason,
                "stage": stage_used,
                "score": (None if not math.isfinite(stage0_score)
                          else float(stage0_score)),
                "composition_weight": float(comp_w),
                "family_bonus": float(family_bonus),
                "piezo_e_max": (float(piezo_raw) if piezo_raw is not None else None),
                "piezo_factor": float(piezo_factor),
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
            and (not args.reject_centrosymmetric_post or rr.get("post_symmetry_ok"))
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
            "post_centrosymmetric": n_post_centrosymmetric,
            "post_non_centrosymmetric": n_post_non_centrosymmetric,
            "post_symmetry_failed": n_post_sym_fail,
            "reject_centrosymmetric_post": bool(args.reject_centrosymmetric_post),
            "ok_fraction": round(n_ok / max(len(cand), 1), 4),
            "converged_ml_fraction_of_ok": round(n_conv_ml / max(n_ok, 1), 4),
            "converged_strict_fraction_of_ok": round(n_conv_strict / max(n_ok, 1), 4),
            "eligible_after_post_novelty": len(eligible),
            "selected": len(selected),
            "ml_thresh_ev_per_A": args.force_converged_ml_thresh,
            "strict_thresh_ev_per_A": args.force_converged_strict_thresh,
            "stage": 1 if args.use_e_form else 0,
            "scarcity_mode": args.scarcity_mode,
            "enrichment_prior": args.enrichment_prior,
            "scarcity_n_weights": len(scarcity_w),
            "scarcity_min_weight": args.scarcity_min_weight,
            "family_prior": args.family_prior,
            "family_bonus_b": args.family_bonus_b,
            "family_bonus_ab": args.family_bonus_ab,
            "family_b_site_Z": list(PIEZO_B_SITE),
            "family_a_site_Z": list(PIEZO_A_SITE),
            "family_with_B_only": n_family_b_only,
            "family_with_AB": n_family_ab,
            "family_with_neither": n_family_neither,
            "piezo_head": (piezo_oracle.config() if piezo_oracle is not None else None),
            "piezo_floor": args.piezo_floor if piezo_oracle is not None else None,
            "piezo_e_max_quantiles": (
                quantiles(piezo_scores) if piezo_scores else None
            ),
            "use_e_form": bool(args.use_e_form),
            "elemental_refs_cache": args.elemental_refs_cache if args.use_e_form else None,
            "e_form_computed_ok": n_e_form_ok,
            "e_form_missing": n_e_form_missing,
            "e_form_coverage_of_ok": (round(n_e_form_ok / max(n_ok, 1), 4)
                                      if args.use_e_form else None),
            "refs_coverage_after_run": (refs.coverage() if refs is not None else None),
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
        # finetune_on default: "relaxed" for chgnet (paper-faithful — train on
        # the oracle's post-relaxation geometry, not the generator's unrelaxed
        # output), "original" otherwise (heuristic path has no relaxed coords).
        finetune_on = args.finetune_on or ("relaxed" if args.oracle == "chgnet" else "original")
        gen.train()
        opt = torch.optim.Adam(gen.parameters(), lr=args.finetune_lr)
        if finetune_on == "relaxed":
            from ..data.representation import Crystal
            crystals = []
            n_fb = 0
            for r in selected:
                rf = r.meta.get("relaxed_frac")
                rl = r.meta.get("relaxed_lattice")
                if rf and rl:
                    crystals.append(Crystal(
                        atom_types=torch.tensor(r.z, dtype=torch.long),
                        frac_coords=torch.tensor(rf, dtype=torch.float),
                        lattice=torch.tensor(rl, dtype=torch.float),
                    ))
                else:
                    crystals.append(r.to_crystal()); n_fb += 1
            print(f"finetune-on={finetune_on}  used relaxed for "
                  f"{len(crystals) - n_fb}/{len(crystals)} "
                  f"(fallback original for {n_fb})", flush=True)
        else:
            crystals = [r.to_crystal() for r in selected]
            print(f"finetune-on={finetune_on}", flush=True)
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

    if args.oracle == "chgnet":
        score_parts = ["(-E_form)" if args.use_e_form else "delta_e",
                       "I_relax_ml", "I_novelty_pre"]
        if args.reject_centrosymmetric_post:
            score_parts.append("I_noncentro")
        if args.scarcity_mode != "none":
            score_parts.append(f"W_comp({args.scarcity_mode})")
        if args.family_prior != "none":
            score_parts.append(f"family({args.family_prior})")
        if args.piezo_head:
            score_parts.append(f"max(|e_max|, {args.piezo_floor})")
        score_label = "Stage-{stage} CHGNet: S = ".format(
            stage=3 if args.piezo_head else (1 if args.use_e_form else 0)
        ) + " * ".join(score_parts)
    else:
        score_label = ("heuristic: volume/atom near target_vpa + min-distance margin"
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
                "reject_centrosymmetric": args.reject_centrosymmetric,
                "reject_centrosymmetric_post": args.reject_centrosymmetric_post,
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
