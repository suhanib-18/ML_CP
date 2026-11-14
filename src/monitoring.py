from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LEVELS = ("NORMAL", "WARNING", "CRITICAL")
PSI_WINDOW = 40
PSI_ALERT = 0.25
PSI_CLEAR = 0.20


@dataclass
class AlertPolicy:
    warning: float
    critical: float
    alpha: float = 0.6
    confirm: int = 2
    release: int = 3
    hysteresis: float = 0.85

    def as_dict(self) -> dict:
        return asdict(self)


class AlertValidator:
    def __init__(self, policy: AlertPolicy):
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self.level = 0
        self.smoothed = None
        self.rising = 0
        self.falling = 0

    def raw_level(self, risk: float) -> int:
        if risk >= self.policy.critical:
            return 2
        if risk >= self.policy.warning:
            return 1
        return 0

    def step(self, risk: float) -> dict:
        policy = self.policy
        self.smoothed = risk if self.smoothed is None else policy.alpha * risk + (1 - policy.alpha) * self.smoothed
        entry = (0.0, policy.warning, policy.critical)
        candidate = self.raw_level(self.smoothed)
        if candidate < self.level and self.smoothed >= entry[self.level] * policy.hysteresis:
            candidate = self.level
        if candidate > self.level:
            self.rising += 1
            self.falling = 0
            if self.rising >= policy.confirm:
                self.level, self.rising = candidate, 0
        elif candidate < self.level:
            self.falling += 1
            self.rising = 0
            if self.falling >= policy.release:
                self.level, self.falling = candidate, 0
        else:
            self.rising = self.falling = 0
        return {
            "risk": float(risk),
            "smoothed": float(self.smoothed),
            "raw_level": self.raw_level(risk),
            "level": self.level,
            "rising": self.rising,
            "falling": self.falling,
        }


def validate_series(risk: np.ndarray, policy: AlertPolicy) -> tuple[np.ndarray, np.ndarray]:
    validator = AlertValidator(policy)
    steps = [validator.step(float(value)) for value in risk]
    return (
        np.array([step["level"] for step in steps], dtype=int),
        np.array([step["smoothed"] for step in steps], dtype=float),
    )


def raw_levels(risk: np.ndarray, policy: AlertPolicy) -> np.ndarray:
    risk = np.asarray(risk, dtype=float)
    return np.where(risk >= policy.critical, 2, np.where(risk >= policy.warning, 1, 0))


def alert_episodes(levels: np.ndarray) -> list[tuple[int, int]]:
    episodes, start = [], None
    for index, active in enumerate(np.append(np.asarray(levels) >= 1, False)):
        if active and start is None:
            start = index
        if not active and start is not None:
            episodes.append((start, index - 1))
            start = None
    return episodes


def event_metrics(levels: np.ndarray, failures: np.ndarray, horizon: int) -> dict:
    levels = np.asarray(levels)
    failures = np.asarray(failures)
    alert = levels >= 1
    failure_index = np.flatnonzero(failures == 1)
    warned, leads = 0, []
    for cycle in failure_index:
        window = alert[max(0, cycle - horizon):cycle]
        hit = bool(window.any())
        warned += hit
        if hit:
            leads.append(len(window) - int(np.argmax(window)))
    episodes = alert_episodes(levels)
    false_episodes = sum(1 for start, end in episodes if not failures[start + 1:end + horizon + 1].any())
    return {
        "failure_events": int(len(failure_index)),
        "warned": int(warned),
        "warned_rate": float(warned / len(failure_index)) if len(failure_index) else 0.0,
        "mean_lead_cycles": float(np.mean(leads)) if leads else 0.0,
        "alert_episodes": int(len(episodes)),
        "false_episodes": int(false_episodes),
        "state_changes": int((np.diff(levels) != 0).sum()),
        "alert_cycles": int(alert.sum()),
        "alert_share": float(alert.mean()) if len(alert) else 0.0,
    }


def quantile_edges(values: np.ndarray, bins: int = 4) -> np.ndarray:
    inner = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, bins + 1)[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def population_stability_index(window, edges: np.ndarray, expected: np.ndarray) -> float:
    if len(window) < PSI_WINDOW:
        return 0.0
    counts, _ = np.histogram(np.asarray(window[-PSI_WINDOW:], dtype=float), bins=np.asarray(edges, dtype=float))
    actual = counts.astype(float) + 0.5
    actual /= actual.sum()
    reference = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    reference /= reference.sum()
    return float(np.sum((actual - reference) * np.log(actual / reference)))
