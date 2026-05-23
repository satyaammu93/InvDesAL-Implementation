"""Diversity sampling for the pretraining corpus.

Paper (Methods / Active learning): "we systematically collected stable crystal
materials spanning different space groups, chemical compositions, and
functional categories ... a diversity sampling strategy to cover different
regions of the inorganic materials distribution."

Reconstruction (REBUILD_PLAN.md sec. 11): the paper states the *objective*
(flatten coverage over space-group / composition regions) but not the exact
rule. We bucket every structure by

    (space group, anonymized stoichiometry, #elements)

and select by **round-robin across buckets**: cycle the buckets, drawing one
random unused member per pass until the target size is reached. This upsamples
rare regions and downsamples abundant ones -> maximal coverage, which is the
stated goal. Inverse-frequency weighted sampling is offered as an alternative.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from typing import Sequence

from .datasets import StructureRecord


def bucket_key(r: StructureRecord) -> tuple:
    cnt = Counter(r.z)
    g = reduce(gcd, cnt.values()) if cnt else 1
    anon_stoich = tuple(sorted(n // g for n in cnt.values()))  # ABX3 -> (1,1,3)
    sg = r.spacegroup if r.spacegroup is not None else -1
    return (sg, anon_stoich, len(cnt))


class DiversitySampler:
    def __init__(self, mode: str = "round_robin", seed: int = 0):
        assert mode in ("round_robin", "inverse_freq")
        self.mode = mode
        self.rng = random.Random(seed)

    def select(
        self, records: Sequence[StructureRecord], target_size: int | None = None
    ) -> list[StructureRecord]:
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            buckets[bucket_key(r)].append(i)

        n_total = len(records)
        target = min(target_size or n_total, n_total)

        if self.mode == "inverse_freq":
            weights = [1.0 / len(buckets[bucket_key(r)]) for r in records]
            idx = self._weighted_sample_without_replacement(
                list(range(n_total)), weights, target
            )
            return [records[i] for i in idx]

        # round-robin: cycle buckets, one random member per visit
        order = list(buckets.keys())
        self.rng.shuffle(order)
        pools = {k: self.rng.sample(v, len(v)) for k, v in buckets.items()}
        chosen: list[int] = []
        while len(chosen) < target:
            progressed = False
            for k in order:
                if pools[k]:
                    chosen.append(pools[k].pop())
                    progressed = True
                    if len(chosen) >= target:
                        break
            if not progressed:
                break
        return [records[i] for i in chosen]

    def _weighted_sample_without_replacement(self, items, weights, k):
        # Efraimidis-Spirakis: key = u^(1/w), take top-k
        keyed = sorted(
            zip(items, weights),
            key=lambda iw: self.rng.random() ** (1.0 / max(iw[1], 1e-12)),
            reverse=True,
        )
        return [it for it, _ in keyed[:k]]

    @staticmethod
    def coverage_report(records: Sequence[StructureRecord]) -> dict:
        b = Counter(bucket_key(r) for r in records)
        sgs = {k[0] for k in b}
        return {
            "n_records": len(records),
            "n_buckets": len(b),
            "n_spacegroups": len(sgs - {-1}),
            "largest_bucket": max(b.values()) if b else 0,
            "singleton_buckets": sum(1 for v in b.values() if v == 1),
        }
