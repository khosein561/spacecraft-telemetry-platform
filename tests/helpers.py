import numpy as np

START_US = 1_767_225_600_000_000  # 2026-01-01T00:00:00Z


def signal(n: int, seed: int = 0) -> np.ndarray:
    """A smooth float32-exact signal (so values survive the 32-bit wire format unchanged)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(t / 15.0) * 0.5 + rng.normal(0, 0.02, n)).astype(np.float32).astype(float)
