# Scoring

A [`Scorer`](/api/#inspect_robots.scorer.Scorer) maps a recorded
[`TrialRecord`](/api/#inspect_robots.rollout.TrialRecord) (plus the scene's
[`Target`](/api/#inspect_robots.scene.Target)) to a [`Score`](/api/#inspect_robots.scorer.Score). Scorers
read the *recorded* trajectory (never a live environment), so scoring is
reproducible from a saved log.

## Builtin scorers

```python
from inspect_robots.scorer import (
    success_at_end,        # 1.0 iff the episode terminated with reason "success"
    episode_length,        # number of steps taken
    min_distance_to_goal,  # closest the effector got (reads StepResult.info["distance"])
    reached_goal_state,    # success iff min distance <= threshold
    operator_scorer,       # reads a human verdict recorded during the rollout
)
```

## Custom scorers

A scorer is any object with a `name` and a `__call__(record, target) -> Score`:

```python
from dataclasses import dataclass
from inspect_robots.scorer import Score

@dataclass(frozen=True)
class SmoothMotion:
    name: str = "smooth_motion"

    def __call__(self, record, target) -> Score:
        deltas = [abs(float(s.action.data.sum())) for s in record.steps]
        return Score(value=-sum(deltas), explanation="negative total command magnitude")
```

Register it with [`scorer`](/api/#inspect_robots.registry.scorer) to resolve it by name.

## Epochs and reducers

When a `Task` runs `epochs > 1`, an epoch reducer collapses the per-epoch
scores of a scene before metrics aggregate across scenes. Reducers are namespaced
separately from metrics and are selected by name on
[`Epochs`](/api/#inspect_robots.task.Epochs):

| Reducer | Meaning |
|---|---|
| `mean`, `median`, `max`, `min` | numeric reductions (raise on non-numeric strings) |
| `mode` | most common value (works for categorical scores) |
| `pass_at_<k>` | unbiased pass@k estimator (success = value ≥ 0.5) |

```python
from inspect_robots.task import Epochs, Task
Task(..., epochs=Epochs(count=5, reducer="pass_at_2"))
```

## Uncertainty and comparing runs

A run's `results.metrics` is a mean per scorer. On a robot that mean usually rests on a few dozen
trials from a handful of scenes, so on its own it cannot say whether a second run that scored
higher is better. [`inspect_robots.evidence`](/api/#inspect_robots.evidence) reads any saved log,
old or new, and adds the missing pieces.

```python
from inspect_robots import compare_logs, metric_evidence, read_eval_log

a = read_eval_log("logs/policy_a.json")
b = read_eval_log("logs/policy_b.json")

ev = metric_evidence(a)["success"]
print(ev.mean, ev.ci_low, ev.ci_high, ev.coverage)   # scene-clustered 95% interval

c = compare_logs(a, b, "success")
print(c.verdict, c.delta, c.wins, c.losses, c.p_permutation, c.mde)
print(c.scenes_needed(0.05))                          # plan the next run
```

Three rules are built in:

- **Scenes are the unit:** epochs of one scene share a world, so intervals resample whole scenes
  and tests permute whole scenes. Adding epochs does not narrow an interval that more scenes would.
- **Coverage travels with the number:** errored trials are never scored, so a mean can be a mean
  over survivors. Every summary reports scored against attempted trials, and a comparison names no
  winner when either side is below `min_coverage` (0.95 by default).
- **Pair by scene:** two runs of the same task see the same scenes, so the comparison is made
  scene by scene. The win, loss and tie counts show when one hard scene is carrying a difference.

The number of scenes also sets a floor on significance that no data can cross: a paired test over
`n` scenes cannot return a two-sided p below `2 / 2**n`. At 5 scenes that is 0.0625, so a 5-scene
comparison cannot separate two policies at 0.05 even when one wins every scene.
`evidence.scenes_to_reach(alpha)` gives the minimum, and `compare_logs` warns when a comparison is
below it. See [`inspect-robots compare`](cli.md#inspect-robots-compare) for the command-line form.

## Operator and VLM scoring (real world)

Real robots have no privileged success oracle. The dominant method is a human
verdict, captured *once* per trial and read back by
[`operator_scorer`](/api/#inspect_robots.scorer.operator_scorer), keeping scoring reproducible.
Benchmarks that read `operator_judgement` directly, instead of delegating to
`operator_scorer`, should call
[`is_affirmative_verdict`](/api/#inspect_robots.scorer.is_affirmative_verdict)
rather than restate the vocabulary. It owns the recognized affirmative words
together with the case-folding and whitespace handling around them, so a change
reaches every consumer at once.
Capture is the job of a [`Grader`](/api/#inspect_robots.grader.Grader): a registered
component (`inspect_robots.graders` entry point, `grader` decorator) whose
`grade(record, scene)` runs once per scored trial, after the rollout and
before the scorers, and writes the judgement onto the record. The builtin
`operator` grader prompts the terminal operator, and the builtin `vlm`
grader is the autograder on the same seam: a vision model judges the trial's
first and last frames against a rubric (the reserved
[`VLMScorer`](/api/#inspect_robots.scorer.VLMScorer) interface predates it
and stays a stub, because R6 requires scorers to be pure readers).

Every attended CLI run is graded by default, registered tasks included, so
judgement-reading scorers (the `operator` scorer, or task scorers that fall
back to `operator_judgement`) work with operator-in-the-loop embodiments and
with policies that end their own trials (`done()`/`give_up()`).
`success_at_end` reads only embodiment-detected `"success"` terminations and
scores operator-graded trials as failures; pair attended operator-graded runs
with a judgement-reading scorer instead. From the Python API, pass
`eval(..., grader="operator")` (or any `Grader` object); unattended runs and
`eval()` without a grader stay prompt-free.
