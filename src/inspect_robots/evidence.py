"""Uncertainty and paired comparison for saved eval logs.

A run's ``results.metrics`` is a bare mean per scorer. On a robot that mean usually rests on a few
dozen trials drawn from a handful of scenes, so on its own it cannot say whether a second run that
scored higher is actually better. This module reads a finished :class:`~inspect_robots.log.EvalLog`
and answers two questions without touching the log format:

- :func:`metric_evidence`: how precisely is each metric known, and how many of the trials the run
  set out to score actually produced a score?
- :func:`compare_logs`: given two runs of the same task, is one better, by how much, and how sure?

Three rules are built in, because each one was learned from a real benchmark going wrong:

**Scenes are the unit of evidence, not trials.** Epochs of one scene share a world, so they are
not independent draws. Every interval resamples whole scenes, and every test permutes whole scenes.
Adding epochs narrows nothing that more scenes would not narrow further.

**Coverage travels with every number.** A trial that errors is recorded but never scored, and the
run-level mean averages whatever survived. When the lost trials are not a random subset (a rate
limit, a spend cap, a crash that hits long rollouts first) the survivor mean is biased in a
direction the log cannot reveal. Every summary here reports scored against attempted trials, and a
comparison refuses to call a winner when either side falls below ``min_coverage``.

**Pair by scene.** Two runs of the same task see the same scenes, so the comparison is made scene
by scene rather than mean against mean. A single hard scene can dominate a difference of means
while the two policies disagree on almost nothing else; the paired statistics, the sign count in
particular, show that directly.

NumPy only. Results are deterministic for a given ``seed``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Literal

import numpy as np

from inspect_robots.log import EvalLog

__all__ = [
    "Comparison",
    "MetricEvidence",
    "compare_logs",
    "holm",
    "metric_evidence",
    "min_attainable_p",
    "paired_permutation_p",
    "scenes_to_reach",
    "sign_test",
]

#: Bootstrap resamples used when the caller does not choose.
DEFAULT_N_BOOT = 4000

#: Coverage below which a comparison is reported but no winner is named.
DEFAULT_MIN_COVERAGE = 0.95

#: Largest scene count for which the paired permutation test enumerates every sign pattern
#: exactly (2**16 = 65,536 patterns). Above it the test samples ``n_perm`` patterns.
EXACT_PERMUTATION_MAX_SCENES = 16

Verdict = Literal[
    "a_better",
    "b_better",
    "not_separated",
    "insufficient_coverage",
    "insufficient_scenes",
]


@dataclass(frozen=True)
class MetricEvidence:
    """One scorer's mean with a scene-clustered interval and the trial accounting behind it.

    ``mean`` is the trial-weighted mean over scored trials, the same quantity a reader would
    compute from the log. ``ci_low``/``ci_high`` resample whole scenes and are ``nan`` when fewer
    than two scenes produced a score, since between-scene variation is then unidentifiable.
    ``coverage`` is ``scored_trials / attempted_trials`` for the whole run, so it is the same for
    every scorer of one log.
    """

    scorer: str
    mean: float
    ci_low: float
    ci_high: float
    n_scenes: int
    scored_trials: int
    attempted_trials: int
    alpha: float

    @property
    def coverage(self) -> float:
        """Fraction of attempted trials that produced a score; ``nan`` for an empty run."""
        if self.attempted_trials == 0:
            return float("nan")
        return self.scored_trials / self.attempted_trials


@dataclass(frozen=True)
class Comparison:
    """Log ``a`` against log ``b`` on one scorer, paired scene by scene.

    ``delta`` is ``mean(a) - mean(b)`` over the paired scenes, each scene weighted equally, with a
    scene-resampled interval. ``wins``/``losses``/``ties`` count scenes where ``a`` scored above,
    below, or level with ``b``. ``p_sign`` is the exact two-sided sign test on that count and
    ``p_permutation`` the two-sided paired sign-flip test on the scene differences; ``verdict``
    reads the latter against ``alpha``. ``mde`` is the difference this comparison had an 80%
    chance of detecting at ``alpha``, and :meth:`scenes_needed` turns a target difference into a
    scene count, both from a normal approximation to the paired differences.
    """

    scorer: str
    delta: float
    ci_low: float
    ci_high: float
    n_paired_scenes: int
    wins: int
    losses: int
    ties: int
    p_sign: float
    p_permutation: float
    alpha: float
    verdict: Verdict
    a: MetricEvidence
    b: MetricEvidence
    sd_difference: float
    unpaired_scenes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mde(self) -> float:
        """Smallest difference with 80% power at this comparison's ``alpha`` and scene count."""
        if self.n_paired_scenes < 2 or not math.isfinite(self.sd_difference):
            return float("nan")
        return _z_sum(self.alpha, 0.8) * self.sd_difference / math.sqrt(self.n_paired_scenes)

    def scenes_needed(self, delta: float, *, power: float = 0.8) -> int | None:
        """Paired scenes needed to detect ``delta`` with ``power``, or ``None`` if unestimable.

        Uses this comparison's observed spread of per-scene differences as the planning value, so
        it is a pilot-based estimate: read it as an order of magnitude, not a guarantee.
        """
        if not 0.0 < power < 1.0:
            raise ValueError(f"power must be in (0, 1), got {power!r}")
        if delta == 0.0 or not math.isfinite(delta):
            raise ValueError(f"delta must be finite and non-zero, got {delta!r}")
        if not math.isfinite(self.sd_difference) or self.n_paired_scenes < 2:
            return None
        if self.sd_difference == 0.0:
            return 2
        n = (_z_sum(self.alpha, power) * self.sd_difference / abs(delta)) ** 2
        return max(2, math.ceil(n))


# -- per-log accounting ------------------------------------------------------


def _scene_scores(log: EvalLog, scorer: str) -> dict[str, list[float]]:
    """Per-scene lists of finite per-trial values for ``scorer``, scenes with none omitted."""
    out: dict[str, list[float]] = {}
    for sample in log.samples:
        values = [
            float(epoch[scorer])
            for epoch in sample.epochs
            if scorer in epoch and _is_number(epoch[scorer])
        ]
        if values:
            out[sample.scene_id] = values
    return out


def _trial_accounting(log: EvalLog) -> tuple[int, int]:
    """``(scored, attempted)`` trials: an empty epoch dict is a trial that errored."""
    attempted = sum(len(sample.epochs) for sample in log.samples)
    scored = sum(1 for sample in log.samples for epoch in sample.epochs if epoch)
    return scored, attempted


def _scorers(log: EvalLog) -> list[str]:
    """Every scorer name that appears in any epoch, in first-seen order."""
    seen: dict[str, None] = {}
    for sample in log.samples:
        for epoch in sample.epochs:
            for name, value in epoch.items():
                if _is_number(value):
                    seen.setdefault(name, None)
    return list(seen)


def metric_evidence(
    log: EvalLog,
    scorers: Sequence[str] | None = None,
    *,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, MetricEvidence]:
    """Each scorer's mean with a scene-clustered interval and the run's trial coverage.

    Args:
        log: A finished eval log.
        scorers: Which scorers to summarize; every numeric scorer in the log by default.
        alpha: Two-sided level of the interval (0.05 gives a 95% interval).
        n_boot: Scene-level bootstrap resamples.
        seed: Seed for the bootstrap, so a repeated call returns the same interval.

    Returns:
        ``{scorer: MetricEvidence}`` in the order requested. A scorer with no finite value in the
        log is reported with ``nan`` statistics and ``n_scenes == 0`` rather than omitted.
    """
    _check_alpha(alpha)
    _check_n(n_boot, "n_boot")
    scored, attempted = _trial_accounting(log)
    names = list(scorers) if scorers is not None else _scorers(log)
    rng = np.random.default_rng(seed)
    out: dict[str, MetricEvidence] = {}
    for name in names:
        groups = list(_scene_scores(log, name).values())
        if not groups:
            nan = float("nan")
            out[name] = MetricEvidence(name, nan, nan, nan, 0, scored, attempted, alpha)
            continue
        mean = float(np.mean([v for g in groups for v in g]))
        low, high = _cluster_interval(groups, alpha, n_boot, rng)
        out[name] = MetricEvidence(name, mean, low, high, len(groups), scored, attempted, alpha)
    return out


def _cluster_interval(
    groups: Sequence[Sequence[float]], alpha: float, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile interval of the trial-weighted mean, resampling whole scenes."""
    if len(groups) < 2:
        return float("nan"), float("nan")
    sums = np.array([sum(g) for g in groups], dtype=np.float64)
    counts = np.array([len(g) for g in groups], dtype=np.float64)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


# -- paired comparison -------------------------------------------------------


def compare_logs(
    log_a: EvalLog,
    log_b: EvalLog,
    scorer: str,
    *,
    alpha: float = 0.05,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = 20000,
    seed: int = 0,
) -> Comparison:
    """Compare two runs of the same task on one scorer, pairing their scenes.

    Scenes are matched by ``scene_id``; a scene scored in only one log is listed in
    ``unpaired_scenes`` and left out of every paired statistic. The verdict names a winner only
    when both logs clear ``min_coverage`` and at least two scenes pair.

    Raises:
        ValueError: If the two logs are of different tasks, or an argument is out of range.
    """
    _check_alpha(alpha)
    _check_n(n_boot, "n_boot")
    _check_n(n_perm, "n_perm")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage!r}")
    if log_a.eval.task != log_b.eval.task:
        raise ValueError(
            f"the logs are of different tasks ({log_a.eval.task!r} and {log_b.eval.task!r}); "
            "a paired comparison needs the same scenes on both sides"
        )

    rng = np.random.default_rng(seed)
    ev_a = metric_evidence(log_a, [scorer], alpha=alpha, n_boot=n_boot, seed=seed)[scorer]
    ev_b = metric_evidence(log_b, [scorer], alpha=alpha, n_boot=n_boot, seed=seed)[scorer]
    scenes_a = _scene_scores(log_a, scorer)
    scenes_b = _scene_scores(log_b, scorer)
    paired = [s for s in scenes_a if s in scenes_b]
    unpaired = tuple(sorted(set(scenes_a) ^ set(scenes_b)))
    diffs = np.array(
        [float(np.mean(scenes_a[s])) - float(np.mean(scenes_b[s])) for s in paired],
        dtype=np.float64,
    )

    warnings = list(_comparability_warnings(log_a, log_b))
    if unpaired:
        warnings.append(
            f"{len(unpaired)} scene(s) scored in only one log were left out of the pairing"
        )
    low_coverage = [
        f"{label} scored {ev.scored_trials} of {ev.attempted_trials} trials"
        for label, ev in (("a", ev_a), ("b", ev_b))
        if ev.attempted_trials and ev.coverage < min_coverage
    ]
    warnings.extend(
        f"{msg}, below the {min_coverage:.0%} coverage bar; its mean is over survivors"
        for msg in low_coverage
    )

    n = len(diffs)
    if n >= 1 and min_attainable_p(n) >= alpha:
        warnings.append(
            f"with {n} paired scene(s) the exact test cannot reach p < {alpha:g} whatever the "
            f"data (smallest attainable p is {min_attainable_p(n):.4f}); at least "
            f"{scenes_to_reach(alpha)} scenes are needed before any pair can separate, and the "
            "percentile interval is optimistic at this size"
        )
    wins = int(np.sum(diffs > 0.0))
    losses = int(np.sum(diffs < 0.0))
    ties = n - wins - losses
    if n == 0:
        nan = float("nan")
        return Comparison(
            scorer, nan, nan, nan, 0, 0, 0, 0, 1.0, 1.0, alpha,
            "insufficient_scenes", ev_a, ev_b, nan, unpaired, tuple(warnings),
        )  # fmt: skip

    delta = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = diffs[idx].mean(axis=1)
        ci_low, ci_high = (float(q) for q in np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0]))
    else:
        ci_low = ci_high = float("nan")
    p_sign = sign_test(wins, losses)
    p_perm = paired_permutation_p(diffs, n_perm=n_perm, rng=rng)

    verdict: Verdict
    if low_coverage:
        verdict = "insufficient_coverage"
    elif n < 2:
        verdict = "insufficient_scenes"
    elif p_perm < alpha:
        verdict = "a_better" if delta > 0.0 else "b_better"
    else:
        verdict = "not_separated"
    return Comparison(
        scorer=scorer,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired_scenes=n,
        wins=wins,
        losses=losses,
        ties=ties,
        p_sign=p_sign,
        p_permutation=p_perm,
        alpha=alpha,
        verdict=verdict,
        a=ev_a,
        b=ev_b,
        sd_difference=sd,
        unpaired_scenes=unpaired,
        warnings=tuple(warnings),
    )


def _comparability_warnings(log_a: EvalLog, log_b: EvalLog) -> list[str]:
    """Differences in run conditions that a paired comparison silently depends on."""
    out: list[str] = []
    for label, left, right in (
        ("embodiment", log_a.eval.embodiment, log_b.eval.embodiment),
        ("max_steps", log_a.eval.max_steps, log_b.eval.max_steps),
        ("environment_id", log_a.eval.environment_id, log_b.eval.environment_id),
        ("environment_revision", log_a.eval.environment_revision, log_b.eval.environment_revision),
    ):
        if left != right:
            out.append(f"{label} differs between the runs ({left!r} vs {right!r})")
    epochs_a = {len(s.epochs) for s in log_a.samples}
    epochs_b = {len(s.epochs) for s in log_b.samples}
    if epochs_a != epochs_b:
        out.append(
            f"epochs per scene differ ({sorted(epochs_a)} vs {sorted(epochs_b)}); "
            "scenes are still weighted equally"
        )
    return out


# -- tests and corrections ---------------------------------------------------


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test at p = 0.5; ties must already be excluded."""
    if wins < 0 or losses < 0:
        raise ValueError(f"counts must be non-negative, got {wins!r} and {losses!r}")
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_permutation_p(
    diffs: np.ndarray, *, n_perm: int = 20000, rng: np.random.Generator | None = None
) -> float:
    """Two-sided paired sign-flip test on the mean of ``diffs``.

    With at most :data:`EXACT_PERMUTATION_MAX_SCENES` differences every sign pattern is
    enumerated and ``p`` is the exact enumerated fraction, which already counts the observed
    pattern and so needs no add-one. Above that, ``n_perm`` patterns are sampled and the
    add-one correction keeps the sampled test valid.
    """
    values = np.asarray(diffs, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 1.0
    observed = abs(float(values.mean()))
    tol = 1e-12 * max(1.0, observed)
    if n <= EXACT_PERMUTATION_MAX_SCENES:
        patterns = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        signs = 1.0 - 2.0 * patterns
        stats = np.abs((signs * values).mean(axis=1))
        return float(np.mean(stats >= observed - tol))
    gen = rng if rng is not None else np.random.default_rng(0)
    signs = gen.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    stats = np.abs((signs * values).mean(axis=1))
    hits = int(np.sum(stats >= observed - tol))
    return (hits + 1) / (n_perm + 1)


def min_attainable_p(n_scenes: int) -> float:
    """Smallest two-sided p the paired sign-flip or sign test can return on ``n_scenes`` scenes.

    Every scene agreeing in sign is the most extreme outcome, and it and its mirror are two of the
    ``2**n`` equally likely sign patterns under the null, so the floor is ``2 / 2**n``. It is a
    property of the design, known before any data is collected: at 5 scenes it is 0.0625, above
    0.05, so a 5-scene comparison cannot separate two policies however far apart they are.
    """
    if n_scenes < 1:
        raise ValueError(f"n_scenes must be >= 1, got {n_scenes!r}")
    return min(1.0, 2.0 / 2.0**n_scenes)


def scenes_to_reach(alpha: float) -> int:
    """Fewest paired scenes whose :func:`min_attainable_p` is below ``alpha``."""
    _check_alpha(alpha)
    n = 1
    while min_attainable_p(n) >= alpha:
        n += 1
    return n


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed as given; controls family-wise error at alpha."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


# -- helpers -----------------------------------------------------------------


def _z_sum(alpha: float, power: float) -> float:
    """``z(1 - alpha/2) + z(power)`` for a two-sided test."""
    unit = NormalDist()
    return unit.inv_cdf(1.0 - alpha / 2.0) + unit.inv_cdf(power)


def _is_number(value: object) -> bool:
    """A finite int or float that is not a bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_n(n: int, name: str) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {n!r}")
