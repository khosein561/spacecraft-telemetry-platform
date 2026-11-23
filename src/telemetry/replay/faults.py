"""
Fault injection for the packet stream.

WHY THIS FILE EXISTS: gap detection needs gaps to find, and you can only claim it works if you know
exactly what was injected. `inject_faults` damages a clean packet stream in four ways and returns
the exact ground truth, so tests can assert "detected == injected".

Faults (each original packet gets at most ONE, chosen at random with the given rates):
  drop     the packet is removed (never arrives)
  corrupt  one random bit is flipped (the CRC must catch it; to the sequence tracker it looks lost)
  dup      the packet is delivered twice (second copy 0-3 positions later)
  reorder  the packet swaps places with its neighbour (only if that neighbour is fault-free)
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultPlan:
    seed: int = 0
    drop_rate: float = 0.0
    corrupt_rate: float = 0.0
    dup_rate: float = 0.0
    reorder_rate: float = 0.0

    def __post_init__(self):
        rates = (self.drop_rate, self.corrupt_rate, self.dup_rate, self.reorder_rate)
        if any(r < 0 for r in rates) or sum(rates) > 1:
            raise ValueError("fault rates must be >= 0 and sum to at most 1")


@dataclass(frozen=True)
class FaultTruth:
    sent: int
    dropped: int
    corrupted: int
    duplicated: int
    reordered: int  # number of swapped PAIRS

    @property
    def never_delivered(self) -> int:
        """Packets whose reading cannot reach the database: dropped + corrupted."""
        return self.dropped + self.corrupted


def inject_faults(packets: list, plan: FaultPlan):
    """Return (damaged_stream, FaultTruth). Pure function of (packets, plan): same input, same output."""
    rng = random.Random(plan.seed)
    n = len(packets)
    labels = []
    for _ in range(n):
        r = rng.random()
        if r < plan.drop_rate:
            labels.append("drop")
        elif r < plan.drop_rate + plan.corrupt_rate:
            labels.append("corrupt")
        elif r < plan.drop_rate + plan.corrupt_rate + plan.dup_rate:
            labels.append("dup")
        elif r < plan.drop_rate + plan.corrupt_rate + plan.dup_rate + plan.reorder_rate:
            labels.append("reorder")
        else:
            labels.append("none")

    # resolve reorders: only valid if the next packet is fault-free (so the swap is unambiguous)
    swapped = set()
    for i in range(n):
        if labels[i] == "reorder":
            if i + 1 < n and labels[i + 1] == "none" and i not in swapped:
                swapped.add(i)
                labels[i + 1] = "partner"
            else:
                labels[i] = "none"

    out = []
    pending_dups = []  # (due_position, bytes)
    i = 0
    while i < n:
        lab = labels[i]
        emitted = []
        if lab == "drop":
            pass
        elif lab == "corrupt":
            damaged = bytearray(packets[i])
            bit = rng.randrange(len(damaged) * 8)
            damaged[bit // 8] ^= 1 << (bit % 8)
            emitted.append(bytes(damaged))
        elif lab == "reorder":
            emitted.extend([packets[i + 1], packets[i]])
            i += 1  # the partner has been consumed
        elif lab == "dup":
            emitted.append(packets[i])
            pending_dups.append((len(out) + rng.randint(1, 4), packets[i]))
        else:
            emitted.append(packets[i])
        out.extend(emitted)
        due = [d for d in pending_dups if d[0] <= len(out)]
        for d in due:
            out.append(d[1])
            pending_dups.remove(d)
        i += 1
    out.extend(b for _, b in pending_dups)  # any duplicates still waiting at end of stream

    truth = FaultTruth(
        sent=n,
        dropped=labels.count("drop"),
        corrupted=labels.count("corrupt"),
        duplicated=labels.count("dup"),
        reordered=len(swapped),
    )
    return out, truth
