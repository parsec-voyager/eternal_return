"""Loops by whole powers of a map's factor, the rounding every exact map shares (sphere.py and surface.py by powers of
2, substitution.py by its rule's): its levels are only the same a whole power apart, so a configured loop factor is
taken as the power nearest it."""

import math

LOG_2 = math.log(2.0)


def nearest_power(factor, log_base=LOG_2):
    """The whole power of the base (given by its log) nearest factor, never 0: a factor over 1 dives at least once, one
    under 1 climbs at least once"""
    power = round(math.log(factor) / log_base) if factor > 0 else 0
    if power == 0:
        power = 1 if factor > 1.0 else -1
    return power


def limited_power(factor, n, room=None, log_base=LOG_2):
    """nearest_power(), climbing no higher than level 0 from level n, and diving at most room levels (None: no limit).
    0: no loop is left that way."""
    power = nearest_power(factor, log_base)
    if power < 0:
        return max(power, -n)
    return power if room is None else max(0, min(power, room))
