import random
import secrets


class RandomSource:
    """Default uses SystemRandom; seeded mode is for experiments only."""

    def __init__(self, seed=None):
        self.rng = secrets.SystemRandom() if seed is None else random.Random(seed)

    def ternary(self, n, positive, negative):
        if min(positive, negative) < 0 or positive + negative > n:
            raise ValueError("Invalid weights")
        a = [1] * positive + [-1] * negative + [0] * (n - positive - negative)
        self.rng.shuffle(a)
        return tuple(a)
