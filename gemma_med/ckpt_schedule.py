"""Log-spaced checkpointing for the interp-trajectory study.

Representational change is fastest in the first few hundred optimizer steps, so
uniform `save_steps` spends most of its checkpoints where nothing is happening.
This saves at steps {1, 2, 4, 8, ...} (+ the final step): dense early, sparse
late. t=0 is not saved here -- it *is* the base checkpoint, so the probe job
prepends the untouched base model as the trajectory's origin.
"""

from __future__ import annotations

from transformers import TrainerCallback


def log_spaced_steps(max_steps: int, base: float = 2.0, include_final: bool = True) -> set[int]:
    """{round(base**k)} for k=0,1,... capped at max_steps, plus max_steps."""
    if max_steps <= 0:
        return set()
    steps: set[int] = set()
    k = 0
    while k <= 4096:
        s = max(1, int(round(base**k)))
        if s >= max_steps:
            break
        steps.add(s)
        k += 1
    if include_final:
        steps.add(max_steps)
    steps.discard(0)
    return steps


class LogSpacedCheckpointCallback(TrainerCallback):
    """Force a save at each log-spaced step. Pair with save_strategy='no' so the
    default uniform saver stays out of the way."""

    def __init__(self, base: float = 2.0, include_final: bool = True) -> None:
        self.base = base
        self.include_final = include_final
        self._steps: set[int] = set()

    def on_train_begin(self, args, state, control, **kwargs):
        self._steps = log_spaced_steps(state.max_steps, self.base, self.include_final)
        # Under an LR-warmup schedule the first optimizer step applies lr=0, so
        # checkpoint-1 is bit-identical to the t=0 base the probe already prepends
        # (verified: 1b-it checkpoint-1 == base, max|Δ|=0). Drop it as a redundant
        # save. A no-warmup run has a meaningful step 1, so keep it there.
        if args.get_warmup_steps(state.max_steps) >= 1:
            self._steps.discard(1)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self._steps:
            control.should_save = True
        return control
