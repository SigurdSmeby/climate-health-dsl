# Concepts: how the disease signal is built

Understanding-oriented — the model behind `disease_cases`, not a field-by-field lookup (that's the [reference](REFERENCE.md)) and not a walkthrough (that's the [tutorial](TUTORIAL.md)).

## How `disease_cases` is generated

The disease signal is a population-relative incidence model, not a plain weighted sum. It builds a per-period incidence *rate*, then draws integer counts from it (Poisson by default, or overdispersed negative binomial via `count_distribution`):

1. Start from a seasonal baseline (one sine cycle per year), so disease has its own seasonality even with no drivers.
2. For each `depends_on` entry: delay the driver by `lag` (causally — the warm-up becomes NaN; values never wrap around from the end), apply any `transforms`, standardize to a z-score, multiply by `weight`, and add.
3. If `autoregressive`, add a random walk (cumulative white noise).
4. Squash through a sigmoid shifted so a typical period lands near `median_rate`, then scale to a rate: `sigmoid × population × max_rate`. The sigmoid guarantees incidence stays below `max_rate` no matter how extreme the drivers get.
5. Draw integer counts from the rate (seeded, per the chosen distribution), capped at `population`.
6. Blank any period with no valid driver signal — the lag warm-up, plus rows where a driver value was itself missing — then apply `missing_rate` last.

See `examples/overdispersed_outbreaks.yaml` for the negative-binomial counts.

## See also

- **[Reference](REFERENCE.md)** — the exact fields this model reads (`lag`, `weight`, `max_rate`, `median_rate`, `count_distribution`, `overdispersion`, …).
- **[Tutorial](TUTORIAL.md)** — see the warm-up and lag effects live, in a running example.
- **[How-to guides](HOW_TO.md)** — add a new generator or transform.
