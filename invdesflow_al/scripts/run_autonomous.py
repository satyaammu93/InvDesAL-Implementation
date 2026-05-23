"""Autonomous orchestrator (~5-6 h, unattended).

Phase 0  verify the bounded lattice-x0 head:
           - Gate 8: one-crystal overfit + sample
           - Gate 9: 32-crystal overfit + 256 samples
         PROCEED to pretraining only if the bound did NOT degrade diversity or
         median lattice quality (unique_rate >= 0.5, 5 <= vpa_median <= 100,
         no NaN). The max volume/atom is NOT a gate -- a bounded-but-imperfect
         tail is expected and shrinks with real training.
Phase 1  small pretrain on 1k diversity-sampled structures + eval.
Phase 2  pretrain on 10k diversity-sampled structures + eval.

Each step logs to logs/<step>.log; orchestrator timeline to logs/autonomous.log;
final summary to logs/autonomous_summary.txt.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/satya/projects/invdes/InvDesFlow-AL")
PY = sys.executable
LOGS = ROOT / "logs"
SUMMARY: list[str] = []


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def note(msg: str) -> None:
    SUMMARY.append(msg)
    log(msg)


def run(args: list[str], logname: str, timeout: float | None = None):
    """Run `python -m <args>` with output to logs/<logname>; return (text, rc)."""
    logpath = LOGS / logname
    log(f"$ python -m {' '.join(args)}   -> logs/{logname}")
    t0 = time.time()
    rc = 0
    with open(logpath, "w") as f:
        try:
            rc = subprocess.run([PY, "-m"] + args, cwd=ROOT, stdout=f,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -1
            f.write("\n[orchestrator] TIMEOUT\n")
    log(f"  done (exit {rc}, {time.time() - t0:.0f}s)")
    return logpath.read_text(errors="ignore"), rc


def result_json(text: str):
    for line in reversed(text.splitlines()):
        if line.startswith("RESULT_JSON "):
            try:
                return json.loads(line[len("RESULT_JSON "):])
            except Exception:
                return None
    return None


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    log("=== autonomous run start ===")

    # ---- Phase 0: verify bounded lattice-x0 head --------------------------
    g8_txt, _ = run(["invdesflow_al.scripts.debug_overfit_one", "--steps", "1500",
                     "--sample", "--device", "cuda"], "auto_gate8.log", 3000)
    g9_txt, _ = run(["invdesflow_al.scripts.debug_tiny_dataset", "--manifest",
                     "data_raw/pretrain.jsonl", "--k", "32", "--steps", "2500",
                     "--n-sample", "256", "--device", "cuda"], "auto_gate9.log", 3600)
    g8, g9 = result_json(g8_txt), result_json(g9_txt)
    note(f"Gate 8 (bounded head): {g8}")
    note(f"Gate 9 (bounded head): {g9}")

    proceed = False
    if g8 and g9:
        ok_div = g9["unique_rate"] >= 0.5
        ok_med = 5.0 <= g9["vpa_median"] <= 100.0
        ok_nan = g9["nan"] == 0
        ok_g8 = bool(g8.get("pass")) or 5.0 <= g8.get("vpa_median", 0) <= 100.0
        proceed = ok_div and ok_med and ok_nan and ok_g8
        note(f"decision: diversity_ok={ok_div} median_ok={ok_med} "
             f"nan_ok={ok_nan} gate8_ok={ok_g8}  ->  proceed={proceed}")
    else:
        note("decision: could not parse gate results -> NOT proceeding")

    if not proceed:
        note("STOP: bounded-head verification failed the proceed criteria. "
             "No pretrain launched. See logs/auto_gate8.log, auto_gate9.log.")
        _write_summary()
        return

    # ---- Phase 1: 1k pretrain --------------------------------------------
    run(["invdesflow_al.scripts.build_manifest", "--inputs", "data_raw/pretrain.jsonl",
         "--max-atoms", "20", "--target-size", "1000", "--seed", "1",
         "--out", "data_raw/pretrain_1k.jsonl"], "auto_build_1k.log", 600)
    run(["invdesflow_al.scripts.train_generator", "--manifest",
         "data_raw/pretrain_1k.jsonl", "--device", "cuda", "--auto-batch",
         "--workers", "0", "--max-hours", "0.6", "--ckpt", str(ROOT / "gen_1k.ckpt"),
         "--log-every", "200", "--ckpt-every", "500"], "auto_train_1k.log", 1.2 * 3600)
    e1_txt, _ = run(["invdesflow_al.scripts.eval_unique_rate", "--ckpt",
                     str(ROOT / "gen_1k.ckpt"), "--manifest", "data_raw/pretrain_1k.jsonl",
                     "--max-samples", "1000", "--gen-batch", "256", "--device", "cuda",
                     "--out", str(ROOT / "eval_1k.json")], "auto_eval_1k.log", 1.5 * 3600)
    note(f"1k pretrain eval: {result_json(e1_txt)}")

    # ---- Phase 2: 10k pretrain -------------------------------------------
    run(["invdesflow_al.scripts.build_manifest", "--inputs", "data_raw/pretrain.jsonl",
         "--max-atoms", "20", "--target-size", "10000", "--seed", "1",
         "--out", "data_raw/pretrain_10k.jsonl"], "auto_build_10k.log", 900)
    run(["invdesflow_al.scripts.train_generator", "--manifest",
         "data_raw/pretrain_10k.jsonl", "--device", "cuda", "--auto-batch",
         "--workers", "0", "--max-hours", "3.5", "--ckpt", str(ROOT / "gen_10k.ckpt"),
         "--log-every", "200", "--ckpt-every", "1000"], "auto_train_10k.log", 4.2 * 3600)
    e10_txt, _ = run(["invdesflow_al.scripts.eval_unique_rate", "--ckpt",
                      str(ROOT / "gen_10k.ckpt"), "--manifest", "data_raw/pretrain_10k.jsonl",
                      "--max-samples", "1000", "--gen-batch", "256", "--device", "cuda",
                      "--out", str(ROOT / "eval_10k.json")], "auto_eval_10k.log", 1.5 * 3600)
    note(f"10k pretrain eval: {result_json(e10_txt)}")
    _write_summary()


def _write_summary() -> None:
    log("=== autonomous run done ===")
    (LOGS / "autonomous_summary.txt").write_text(
        f"autonomous run summary ({now()})\n\n" + "\n".join(SUMMARY) + "\n"
    )
    log(f"summary -> logs/autonomous_summary.txt")


if __name__ == "__main__":
    main()
