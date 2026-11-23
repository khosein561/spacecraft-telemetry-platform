"""
Sequence-count tracking: detect loss, duplication and reordering on a packet stream.

WHY THIS FILE EXISTS: a spacecraft link drops, repeats and reorders packets. The 14-bit sequence
count (one counter per APID) is how the ground notices. Two things make this subtle:

  1. Wraparound: 16383 -> 0 is the NORMAL next packet, not a gap. All arithmetic is modulo 16384.
  2. Direction: with only 14 bits you cannot know whether seq 5 after seq 9 is "4 behind" or
     "16380 ahead". Convention used here: a forward distance of less than half the space (8192)
     is "ahead" (a gap); anything else is "behind" (a repeat or a late arrival).

The tracker also remembers which sequence numbers it skipped, so a packet that arrives late and
fills a hole is recognised as REORDERING, not as a new fault, and the final `lost` count is the
number of packets that never arrived at all.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .ccsds import SEQ_MODULO

HALF = SEQ_MODULO // 2
RECENT_WINDOW = 256  # how many recent sequence numbers we remember to recognise repeats
MAX_TRACKED_GAP = 4096  # beyond this we count the loss but stop remembering individual numbers

FIRST = "first"
IN_ORDER = "in_order"
GAP = "gap"
DUPLICATE = "duplicate"
OUT_OF_ORDER = "out_of_order"


@dataclass(frozen=True)
class SeqEvent:
    kind: str
    gap: int = 0  # number of packets skipped (only for kind == GAP)
    filled_gap: bool = False  # True when an out-of-order packet filled a known hole


@dataclass
class SequenceTracker:
    """Tracks ONE stream (one APID). Use `StreamTrackers` for many."""

    last: int | None = None
    seen_recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    missing: set = field(default_factory=set)
    gaps: int = 0  # number of gap events
    skipped: int = 0  # total packets skipped over by gap events (before late arrivals fill holes)
    duplicates: int = 0
    out_of_order: int = 0
    untracked_lost: int = 0  # losses in gaps too large to remember individually
    received: int = 0

    def observe(self, seq: int) -> SeqEvent:
        if not 0 <= seq < SEQ_MODULO:
            raise ValueError(f"sequence count out of range: {seq}")
        self.received += 1
        if self.last is None:
            self.last = seq
            self.seen_recent.append(seq)
            return SeqEvent(FIRST)

        expected = (self.last + 1) % SEQ_MODULO
        ahead = (seq - expected) % SEQ_MODULO

        if ahead == 0:
            self.last = seq
            self.seen_recent.append(seq)
            self.missing.discard(seq)
            return SeqEvent(IN_ORDER)

        if ahead < HALF:  # jumped forward: `ahead` packets were skipped
            if ahead <= MAX_TRACKED_GAP:
                for i in range(ahead):
                    self.missing.add((expected + i) % SEQ_MODULO)
            else:
                self.untracked_lost += ahead
            self.gaps += 1
            self.skipped += ahead
            self.last = seq
            self.seen_recent.append(seq)
            return SeqEvent(GAP, gap=ahead)

        # seq is behind `last`: a repeat or a late arrival
        if seq in self.missing:
            self.missing.discard(seq)
            self.seen_recent.append(seq)
            self.out_of_order += 1
            return SeqEvent(OUT_OF_ORDER, filled_gap=True)
        if seq in self.seen_recent:
            self.duplicates += 1
            return SeqEvent(DUPLICATE)
        self.out_of_order += 1  # behind us, never seen, not a known hole (e.g. after a long outage)
        return SeqEvent(OUT_OF_ORDER)

    @property
    def lost(self) -> int:
        """Packets that never arrived (holes still open + losses too big to track)."""
        return len(self.missing) + self.untracked_lost


class StreamTrackers:
    """One SequenceTracker per APID (sequence counts are per-APID in CCSDS)."""

    def __init__(self) -> None:
        self._by_apid: dict[int, SequenceTracker] = {}

    def observe(self, apid: int, seq: int) -> SeqEvent:
        return self._by_apid.setdefault(apid, SequenceTracker()).observe(seq)

    def tracker(self, apid: int) -> SequenceTracker:
        return self._by_apid.setdefault(apid, SequenceTracker())

    def totals(self) -> dict:
        t = self._by_apid.values()
        return {
            "received": sum(x.received for x in t),
            "gaps": sum(x.gaps for x in t),
            "lost": sum(x.lost for x in t),
            "duplicates": sum(x.duplicates for x in t),
            "out_of_order": sum(x.out_of_order for x in t),
        }
