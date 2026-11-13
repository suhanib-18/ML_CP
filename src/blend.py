"""Blend used by the horizon stacking core.

Lives in its own module so the commissioned core can be reloaded by the console.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

IGNORE_FRACTION = 0.55


class OpeningBlockLogisticRegression(LogisticRegression):
    """Balanced logistic blend fit on recent cycles only.

    Earlier cycles are a different operating regime, and the opening block
    has no past-only out-of-fold score. Both are left out of the blend.
    """

    def fit(self, X, y, sample_weight=None):
        count = len(y)
        weights = np.ones(count, dtype=float)
        if sample_weight is not None:
            weights *= np.asarray(sample_weight, dtype=float)
        weights[: int(count * IGNORE_FRACTION)] = 0.0
        return super().fit(X, y, sample_weight=weights)
