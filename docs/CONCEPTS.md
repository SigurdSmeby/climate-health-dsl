# Concepts: how the disease signal is built

Understanding-oriented — the model behind `disease_cases`, not a field-by-field lookup (that's the [reference](REFERENCE.md)) and not a walkthrough (that's the [tutorial](TUTORIAL.md)).

## How `disease_cases` is generated

The disease signal is a population-relative incidence model, not a plain weighted sum. It builds a per-period incidence *rate*, then draws integer counts from it (Poisson by default, or overdispersed negative binomial via `distribution`):

1. Start from the series' own generator, or from flat 0 when it has only parents.
2. For each `depends_on` entry: delay the parent by `lag` (causally — the warm-up becomes NaN; values never wrap around from the end), apply any `transforms`, standardize to a z-score, multiply by `weight`, and add.
3. Apply any `events` this series reacts to, before anything downstream reads it.
4. If `autoregressive`, add an AR(1) process: `x[t] = phi·x[t-1] + noise`. Unlike a random walk it stays bounded, and `phi` is itself a declared ground truth a model can be scored on recovering.
5. Squash through a sigmoid shifted so a typical period lands near `median_rate`, then scale to a rate: `sigmoid × population × max_rate`. That bounds the *rate*; individual draws still scatter around it, and only `population` hard-caps a drawn value.
6. Draw integer counts from the rate (seeded, per the chosen distribution), capped at `population`.
7. Blank any period with no valid driver signal — the lag warm-up, plus rows where a parent value was itself missing — then apply `missing_rate` last.

There is no built-in seasonal term: a disease is seasonal because its drivers are. That way the seasonal amplitude a model recovers is the one the scenario planted, rather than that plus an invisible constant.

See `examples/overdispersed_outbreaks.yaml` for the negative-binomial counts.

## See also

- **[Reference](REFERENCE.md)** — the exact fields this model reads (`lag`, `weight`, `max_rate`, `median_rate`, `distribution`, `overdispersion`, …).
- **[Tutorial](TUTORIAL.md)** — see the warm-up and lag effects live, in a running example.
- **[How-to guides](HOW_TO.md)** — add a new generator or transform.
