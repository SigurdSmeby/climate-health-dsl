# How to: extend the DSL

Task-oriented guides. Looking for what a field does instead? See the [reference](REFERENCE.md). New to the DSL? Start with the [tutorial](TUTORIAL.md).

## Add a new generator, transform or emitter

One mental model: **a generator creates a series, a transform modifies one, an emitter decides what a finished series becomes in the CSV.** All three live in extension folders where every file registers itself — you never edit the core machinery.

First check whether you need code at all. A new *series* that reuses an existing shape is pure YAML:

```yaml
series:
  - name: wind
    generate: seasonal_smooth  # reuse
    params: { mean: 12, amplitude: 4 }
```

And a whole *scenario* you keep retyping is a [preset](REFERENCE.md#presets-start-from-a-scenario-write-only-the-difference) — a plain scenario YAML dropped in `src/dsl/presets/`, usable as `preset: <filename>` with no code and no registration. `dsl list` picks it up automatically.

A new *shape* is one new file in `src/dsl/generators/`:

```python
"""Gusty wind: a noisy series with occasional sharp spikes."""
import numpy as np
from dsl.core.extension.generator_base import VariableGenerator, register_generator

@register_generator("gusty")  # the name you write in YAML
class GustyGenerator(VariableGenerator):
    def __init__(self, base: float = 5.0, gust_chance: float = 0.1):
        self.base = base  # these are the YAML `params:`
        self.gust_chance = gust_chance

    def generate(self, n_periods, period, rng):
        series = rng.normal(self.base, 1.0, size=n_periods)
        gusts = rng.random(n_periods) < self.gust_chance
        series[gusts] += rng.normal(10, 2, size=gusts.sum())
        return series
```

The file is auto-discovered on import, the schema passes `params` through, and the engine finds the name in the registry — no other file changes.

**Transforms** work identically under `src/dsl/transforms/`: subclass `Transform`, implement `apply(series, rng)`, register with `@register_transform`. A transform becomes usable from a `depends_on[].transforms` list with no core change.

**Emitters** decide a series' output type, under `src/dsl/emitters/`: subclass `Emitter`, implement `emit(signal, rng)`, register with `@register_emitter`. Everything before that final step — generator, parents, events, persistence, gaps — is shared, so an emitter is only the last piece. The registered *name* is the YAML block, so `@register_emitter("binary")` makes `binary: {threshold: 0.2}` a valid series block beside `counts:`.

If a generator plants something at an identifiable period — a seasonal peak, an outbreak — override `events()` to report it. Those periods become exact ground truth, and the fold report counts them instead of guessing from the output.

Note the auto-import loop skips modules whose name starts with `_`, so a scratch file is ignored until renamed.

The only core file ever edited after the initial build is `src/dsl/core/config/schema.py`, and only for a genuinely new top-level concept (a new series field, a new global setting) — with schema tests in the same commit.

See the [reference](REFERENCE.md#generators) for the full list of built-in generators, transforms and emitters to use as more examples, and [CODE_STYLE_GUIDE.md](CODE_STYLE_GUIDE.md) for the docstring/comment conventions new code should follow.

## Add a preset

A preset is an ordinary scenario file in `src/dsl/presets/`, and the filename is the name you write in YAML. Nothing registers it — `available()` globs the folder, so dropping `tropical.yaml` there makes `preset: tropical` work and `dsl list` show it.

Two things to check before adding one:

- **It must run on its own.** `preset: <name>` with nothing else is a complete scenario; there is a test that runs every bundled preset exactly that way.
- **If it sets `split:`, keep it warning-free.** `folds/report.md` warns when a fold's test half cannot measure something the scenario plants, and a preset meant for evaluation should not ship with that warning. The `evaluation` preset uses 120 periods for exactly this reason — fewer leaves fold 1's test half too narrow for the yearly rainfall peak.

## See also

- **[Reference](REFERENCE.md)** — every built-in generator, transform and emitter, as examples.
- **[Concepts](CONCEPTS.md)** — how a transform's output feeds into the disease model.
- **[Tutorial](TUTORIAL.md)** — a hands-on walkthrough if you're starting out.
